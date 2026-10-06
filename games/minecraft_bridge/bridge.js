// flyverse games/minecraft -- the Node side of the bridge.
//
// One process, driven by games/minecraft.py over stdio (one JSON object per line each way):
//   * a mineflayer bot (the fly's body in the game). Its physics runs only when Python says `step`: one
//     prismarine-physics tick (Minecraft's own client movement code) per 50 ms of brain time, so the bot stays in
//     lockstep with a brain that runs slower than real time. The server's world is frozen (/tick freeze) and
//     stepped by Python over RCON; `sync` waits for a chat marker that proves this bot has received that tick.
//   * `region`: the real block grid around the bot (palette names + uint16 indices) for the fly's eyes.
//   * a prismarine-viewer page (three.js, the project's own Minecraft renderer) in headless Chrome, rendered on
//     demand from given cameras (`frame`) and returned as JPEG: the footage is Minecraft's blocks, textures and
//     entity models as prismarine-viewer draws them.
// Nothing here decides anything for the fly. All logging goes to stderr; stdout carries protocol lines only.
'use strict'

const path = require('path')
const fs = require('fs')
const readline = require('readline')

const realLog = console.log
console.log = (...a) => process.stderr.write('[log] ' + a.map(String).join(' ') + '\n')
console.info = console.log
console.warn = (...a) => process.stderr.write('[warn] ' + a.map(String).join(' ') + '\n')
void realLog

const mineflayer = require('mineflayer')
const { Vec3 } = require('vec3')
const { Physics, PlayerState } = require('prismarine-physics')

const err = (...a) => process.stderr.write('[bridge] ' + a.map(String).join(' ') + '\n')
const send = (obj) => process.stdout.write(JSON.stringify(obj) + '\n')

let bot = null
let physics = null
const markers = new Set()
const markerWaiters = new Map()
const chatLog = []
let hurtEvents = []
let viewer = null // { http, io, browser, page }

// ------------------------------------------------------------------------------------------------ bot
function connect ({ host = '127.0.0.1', port = 25631, username = 'FlyBot', version = '1.21.4' }) {
  return new Promise((resolve, reject) => {
    bot = mineflayer.createBot({ host, port, username, version, auth: 'offline', physicsEnabled: false, viewDistance: 'normal' })
    const t = setTimeout(() => reject(new Error('spawn timeout')), 60000)
    bot.once('spawn', () => {
      clearTimeout(t)
      bot.physicsEnabled = false
      physics = Physics(bot.registry, bot.world)
      resolve({ entityId: bot.entity.id, pos: vec(bot.entity.position), yaw: bot.entity.yaw, version: bot.version, attr: physics.movementSpeedAttribute })
    })
    bot.on('messagestr', (msg) => {
      if (msg.startsWith('#fv ')) {
        const k = msg.slice(4).trim()
        markers.add(k)
        const w = markerWaiters.get(k)
        if (w) { markerWaiters.delete(k); w() }
      } else {
        chatLog.push(msg)
        if (chatLog.length > 50) chatLog.shift()
      }
    })
    bot.on('entityHurt', (e) => { if (e === bot.entity) hurtEvents.push({ health: bot.health }) })
    bot.on('death', () => hurtEvents.push({ death: true }))
    bot.on('kicked', (r) => err('kicked', JSON.stringify(r)))
    bot.on('error', (e) => err('bot error', e && e.message))
    bot.on('end', (r) => err('bot end', r))
  })
}

const vec = (v) => [v.x, v.y, v.z]

function botState () {
  const e = bot.entity
  return {
    pos: vec(e.position),
    vel: vec(e.velocity),
    yaw: e.yaw,
    pitch: e.pitch,
    onGround: !!e.onGround,
    collidedH: !!e.isCollidedHorizontally,
    health: bot.health,
    food: bot.food,
    alive: bot.isAlive !== false
  }
}

function step ({ yaw, pitch = 0, forward = false, back = false, jump = false, sprint = false, speed = 0.1 }) {
  // one client physics tick (prismarine-physics = Minecraft's player movement), then tell the server where we are
  const e = bot.entity
  e.yaw = yaw
  e.pitch = pitch
  e.attributes = e.attributes || {}
  e.attributes[physics.movementSpeedAttribute] = { value: speed, modifiers: [] }
  const controls = { forward, back, left: false, right: false, jump, sprint, sneak: false }
  physics.simulatePlayer(new PlayerState(bot, controls), bot.world).apply(bot)
  const onGround = !!e.onGround
  bot._client.write('position_look', {
    x: e.position.x,
    y: e.position.y,
    z: e.position.z,
    yaw: 180 - (e.yaw * 180 / Math.PI), // mineflayer yaw (rad, 0 = north, CCW) -> notchian degrees
    pitch: -(e.pitch * 180 / Math.PI),
    onGround,
    flags: { onGround, hasHorizontalCollision: !!e.isCollidedHorizontally }
  })
  bot.emit('move', e.position)
  return botState()
}

function entities (radius = 64) {
  const me = bot.entity.position
  const out = []
  for (const id in bot.entities) {
    const e = bot.entities[id]
    if (!e || e === bot.entity || !e.position) continue
    if (e.position.distanceTo(me) > radius) continue
    out.push({ id: e.id, name: e.name, type: e.type, username: e.username, pos: vec(e.position), yaw: e.yaw, pitch: e.pitch, width: e.width, height: e.height })
  }
  return out
}

function snapshot () {
  const h = hurtEvents
  hurtEvents = []
  return { bot: botState(), entities: entities(), hurt: h }
}

function waitMarker (k, timeoutMs = 10000) {
  k = String(k)
  if (markers.has(k)) { markers.delete(k); return Promise.resolve(true) }
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => { markerWaiters.delete(k); reject(new Error('marker timeout ' + k)) }, timeoutMs)
    markerWaiters.set(k, () => { clearTimeout(t); markers.delete(k); resolve(true) })
  })
}

function region ({ x0, y0, z0, dx, dy, dz }) {
  // palette indices, order ((y * dz) + z) * dx + x, of every block in the box [x0, x0+dx) x [y0, y0+dy) x [z0, z0+dz)
  const data = new Uint16Array(dx * dy * dz)
  const palette = []
  const pidx = new Map()
  const p = new Vec3(0, 0, 0)
  let missing = 0
  for (let y = 0; y < dy; y++) {
    for (let z = 0; z < dz; z++) {
      for (let x = 0; x < dx; x++) {
        p.set(x0 + x, y0 + y, z0 + z)
        let sid = null
        try { sid = bot.world.getBlockStateId(p) } catch (e) { sid = null }
        let name
        if (sid === null || sid === undefined) { name = 'void_air'; missing++ } else {
          const b = bot.registry.blocksByStateId[sid]
          name = b ? b.name : 'air'
        }
        let i = pidx.get(name)
        if (i === undefined) { i = palette.length; palette.push(name); pidx.set(name, i) }
        data[(y * dz + z) * dx + x] = i
      }
    }
  }
  return { origin: [x0, y0, z0], dims: [dx, dy, dz], palette, missing, data: Buffer.from(data.buffer).toString('base64') }
}

function heightmap ({ x0, z0, dx, dz, ytop = 200, ybot = 40 }) {
  // top non-air block per column and its name (for choosing a scenic start; not used by the fly)
  const heights = new Int16Array(dx * dz)
  const names = []
  const nidx = new Map()
  const tops = new Uint16Array(dx * dz)
  const p = new Vec3(0, 0, 0)
  for (let z = 0; z < dz; z++) {
    for (let x = 0; x < dx; x++) {
      let h = -999
      let name = 'void_air'
      for (let y = ytop; y >= ybot; y--) {
        p.set(x0 + x, y, z0 + z)
        let b = null
        try { b = bot.world.getBlockStateId(p) } catch (e) { b = null }
        if (b === null || b === undefined) break
        const bl = bot.registry.blocksByStateId[b]
        if (bl && bl.name !== 'air' && bl.name !== 'cave_air' && bl.boundingBox === 'block') { h = y; name = bl.name; break }
        if (bl && (bl.name === 'water')) { h = y; name = 'water'; break }
      }
      let i = nidx.get(name)
      if (i === undefined) { i = names.length; names.push(name); nidx.set(name, i) }
      heights[z * dx + x] = h
      tops[z * dx + x] = i
    }
  }
  return { heights: Array.from(heights), tops: Array.from(tops), names }
}

// ------------------------------------------------------------------------------------------------ viewer
const PV = path.join(__dirname, 'node_modules', 'prismarine-viewer')

function patchedIndexJs () {
  let src = fs.readFileSync(path.join(PV, 'public', 'index.js'), 'utf8')
  const rep = [
    ['const l=new THREE.WebGLRenderer;', 'const l=new THREE.WebGLRenderer({antialias:!0,preserveDrawingBuffer:!0});'],
    ['const u=new r(l);', 'const u=new r(l);window.viewer=u;window.fvRenderer=l;window.fvEntity=a;'],
    ['let h=new THREE.OrbitControls(u.camera,l.domElement);', 'let h=null;'],
    ['!function t(){window.requestAnimationFrame(t),h&&h.update(),u.update(),l.render(u.scene,u.camera)}()', 'void 0']
  ]
  for (const [a, b] of rep) {
    if (!src.includes(a)) throw new Error('prismarine-viewer bundle changed; cannot patch: ' + a.slice(0, 40))
    src = src.replace(a, b)
  }
  return src
}

async function viewerStart ({ port = 3031, width = 1536, height = 864, viewDistance = 6, headless = true }) {
  const express = require('express')
  const compression = require('compression')
  const { WorldView } = require(path.join(PV, 'viewer', 'lib', 'worldView'))
  const app = express()
  const http = require('http').createServer(app)
  const io = require('socket.io')(http, { path: '/socket.io' })
  const indexJs = patchedIndexJs()
  app.use(compression())
  app.get('/', (req, res) => res.sendFile(path.join(__dirname, 'web', 'index.html')))
  app.get('/index.js', (req, res) => res.type('application/javascript').send(indexJs))
  app.get('/fv.js', (req, res) => res.sendFile(path.join(__dirname, 'web', 'fv.js')))
  app.use('/', express.static(path.join(PV, 'public')))
  const vs = { worldView: null, pending: Promise.resolve() }
  io.on('connection', (socket) => {
    socket.emit('version', bot.version)
    const wv = new WorldView(bot.world, viewDistance, bot.entity.position, socket)
    vs.worldView = wv
    // the page must have set its version before chunks arrive; give it a moment, then load every column in range
    vs.pending = new Promise((resolve) => setTimeout(resolve, 300)).then(() => wv.init(bot.entity.position))
    wv.listenToBot(bot)
    const onMove = () => { vs.pending = vs.pending.then(() => wv.updatePosition(bot.entity.position)) }
    bot.on('move', onMove)
    socket.on('disconnect', () => { bot.removeListener('move', onMove); wv.removeListenersFromBot(bot) })
  })
  await new Promise((resolve) => http.listen(port, '127.0.0.1', resolve))
  const puppeteer = require('puppeteer')
  const browser = await puppeteer.launch({
    headless: headless ? 'new' : false,
    args: ['--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-webgl', `--window-size=${width},${height}`, '--no-sandbox']
  })
  const page = await browser.newPage()
  page.on('console', (m) => { const t = m.text(); if (!/THREE|Using version|WebGL/.test(t)) err('page:', t.slice(0, 300)) })
  page.on('pageerror', (e) => err('pageerror:', e.message))
  await page.setViewport({ width, height, deviceScaleFactor: 1 })
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load' })
  await page.waitForFunction('window.viewer && window.fvReady && window.viewer.version', { timeout: 60000 })
  viewer = { http, io, browser, page, vs }
  const gl = await page.evaluate(() => window.fvGlInfo())
  return { ok: true, gl }
}

async function viewerCall (fn, arg) {
  if (!viewer) throw new Error('viewer not started')
  return viewer.page.evaluate((f, a) => window[f](a), fn, arg)
}

async function viewerWait (a) {
  // every column in range has been handed to the page, then the page's mesher has caught up (or a timeout)
  if (!viewer) throw new Error('viewer not started')
  await viewer.vs.pending
  const loaded = viewer.vs.worldView ? Object.keys(viewer.vs.worldView.loadedChunks).length : 0
  const r = await viewerCall('fvWait', Object.assign({ expectColumns: loaded }, a))
  r.columns = loaded
  return r
}

async function frame (a) {
  await viewer.vs.pending
  if (a.wait) {
    const loaded = viewer.vs.worldView ? Object.keys(viewer.vs.worldView.loadedChunks).length : 0
    await viewerCall('fvWait', { timeoutMs: a.wait, expectColumns: loaded, settleMs: 0 })
  }
  // every entity's name, so the page can fix the texture mapping of models it draws beyond the exported radius
  a.names = {}
  for (const id in bot.entities) { const e = bot.entities[id]; if (e && e.name) a.names[id] = e.name }
  return viewerCall('fvRender', a)
}

// ------------------------------------------------------------------------------------------------ loop
const handlers = {
  async connect (a) { return connect(a) },
  async step (a) { return step(a) },
  async snapshot () { return snapshot() },
  async sync (a) { await waitMarker(a.k, a.timeoutMs || 10000); return snapshot() },
  async region (a) { return region(a) },
  async heightmap (a) { return heightmap(a) },
  async state () { return botState() },
  async chat () { return chatLog.splice(0) },
  async viewer_start (a) { return viewerStart(a) },
  async viewer_setup (a) { return viewerCall('fvSetup', a) },
  async viewer_wait (a) { return viewerWait(a) },
  async frame (a) { return frame(a) },
  async wait_chunks () { await bot.waitForChunksToLoad(); return Object.keys(bot.world.async.columns || {}).length },
  async columns () { return bot.world.getColumns().length },
  async quit () {
    if (viewer) { try { await viewer.browser.close() } catch (e) {} try { viewer.http.close() } catch (e) {} }
    if (bot) { try { bot.quit() } catch (e) {} }
    setTimeout(() => process.exit(0), 200)
    return true
  }
}

const rl = readline.createInterface({ input: process.stdin })
let chain = Promise.resolve()
rl.on('line', (line) => {
  if (!line.trim()) return
  let msg
  try { msg = JSON.parse(line) } catch (e) { send({ id: null, ok: false, error: 'bad json' }); return }
  chain = chain.then(async () => {
    const h = handlers[msg.cmd]
    if (!h) { send({ id: msg.id, ok: false, error: 'unknown cmd ' + msg.cmd }); return }
    try {
      const result = await h(msg.args || {})
      send({ id: msg.id, ok: true, result })
    } catch (e) {
      send({ id: msg.id, ok: false, error: String(e && e.stack ? e.stack : e) })
    }
  })
})
rl.on('close', () => { if (bot) try { bot.quit() } catch (e) {} process.exit(0) })
process.on('uncaughtException', (e) => err('uncaught', e && e.stack))
send({ id: 0, ok: true, result: 'ready' })
