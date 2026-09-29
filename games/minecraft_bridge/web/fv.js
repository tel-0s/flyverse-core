// flyverse games/minecraft: on-demand rendering of the prismarine-viewer scene from cameras chosen by the game.
/* global THREE */
(function () {
  const v = window.viewer
  const r = window.fvRenderer
  let sky = null
  let sun = null
  let botMesh = null
  let sunDir = new THREE.Vector3(0.9, 0.45, 0).normalize()
  const base = new WeakMap()
  // bone order of the viewer's entity models (entities.json), for a simple limb swing
  const BONES = {
    player: { lLeg: 12, rLeg: 14, lArm: 6, rArm: 9 },
    zombie: { lLeg: 8, rLeg: 7, lArm: 6, rArm: 4, zombieArms: true },
    husk: { lLeg: 8, rLeg: 7, lArm: 6, rArm: 4, zombieArms: true },
    creeper: { quad: [2, 3, 4, 5] }
  }
  // vanilla PhantomRenderer draws the model 1.3125 blocks lower and 0.1875 further back than its model origin
  // (prismarine-viewer omits this), so without it the phantom floats 1.3 m above the hitbox the fly sees
  const PHANTOM_OFFSET = [0, -1.3125, 0.1875]

  // prismarine-viewer's entities.json declares some models' texture size wrongly for the 1.16.4 textures it loads:
  // zombie / husk geometry says 64 x 32 but the skins are 64 x 64 (so every face samples twice too far down and the
  // torso, arms and legs land on transparent texels and vanish), sheep says 64 x 64 for a 64 x 32 texture. The
  // classic 64 x 32 layout is the top half of a 64 x 64 skin, so scaling v fixes it. (Checked against the PNG sizes.)
  const UV_V_SCALE = { zombie: 0.5, husk: 0.5, sheep: 2.0 }

  function skinnedOf (mesh) {
    return mesh.children.filter((c) => c.skeleton)
  }

  function fixUv (m, name) {
    const s = UV_V_SCALE[name]
    if (!s || m.userData.fvUv) return
    for (const c of skinnedOf(m)) {
      const uv = c.geometry.attributes.uv
      for (let i = 0; i < uv.count; i++) uv.setY(i, uv.getY(i) * s)
      uv.needsUpdate = true
    }
    m.userData.fvUv = true
  }

  // Rotate bone `idx` of a skinned mesh by `dx` about the x axis through its JSON pivot. prismarine-viewer's Entity.js
  // sets every bone's position to its ABSOLUTE JSON pivot and then nests the bones, so a bone's origin is the sum of
  // its ancestors' pivots (a zombie arm's origin is 36 px above its shoulder) and a plain `rotation.x` swings the
  // limb about the wrong point. Instead set the bone's local matrix so that, relative to the bind pose, its vertices
  // turn about the true pivot p: chain_new = T(p) Rx(dx) T(-p) chain_bind, local = parent_chain_bind^-1 chain_new.
  // (Ancestors are never posed here, so the parent's chain is its bind chain; the skinned mesh had the identity
  // world matrix at bind time, so chain_bind = boneInverse^-1.)
  function poseBone (sm, idx, dx) {
    const sk = sm.skeleton
    const b = sk.bones[idx]
    if (!b) return
    let st = base.get(b)
    if (!st) {
      const pi = sk.bones.indexOf(b.parent)
      st = {
        pivot: b.position.clone(),
        bind: sk.boneInverses[idx].clone().invert(),
        parentInv: pi >= 0 ? sk.boneInverses[pi].clone() : new THREE.Matrix4()
      }
      base.set(b, st)
    }
    const p = st.pivot
    const m = new THREE.Matrix4().makeTranslation(p.x, p.y, p.z)
      .multiply(new THREE.Matrix4().makeRotationX(dx))
      .multiply(new THREE.Matrix4().makeTranslation(-p.x, -p.y, -p.z))
      .multiply(st.bind)
    new THREE.Matrix4().multiplyMatrices(st.parentInv, m).decompose(b.position, b.quaternion, b.scale)
  }

  function pose (mesh, name, walk, speed) {
    const map = BONES[name]
    if (!map) return
    const amp = Math.min(1, speed / 2.0) * 0.9
    const s = Math.sin(walk)
    for (const sm of skinnedOf(mesh)) {
      const setX = (i, dx) => poseBone(sm, i, dx)
      if (map.quad) {
        setX(map.quad[0], s * amp); setX(map.quad[1], -s * amp); setX(map.quad[2], -s * amp); setX(map.quad[3], s * amp)
        continue
      }
      setX(map.lLeg, s * amp)
      setX(map.rLeg, -s * amp)
      // the model faces -z; +PI/2 about x turns a hanging arm (-y) to point forward (-z): a zombie's raised arms
      if (map.zombieArms) { setX(map.lArm, Math.PI / 2 + 0.08 * s); setX(map.rArm, Math.PI / 2 - 0.08 * s) } else {
        setX(map.lArm, -s * amp * 0.8); setX(map.rArm, s * amp * 0.8)
      }
    }
  }

  function placePhantom (m, pitch) {
    // once: move the drawn model onto its hitbox (vanilla's offset); every frame: pitch the whole model about the
    // entity origin like vanilla's setupRotations (mineflayer pitch: + = nose up)
    if (!m.userData.fvPhantom) {
      for (const c of m.children) if (c.skeleton) c.position.set(PHANTOM_OFFSET[0], PHANTOM_OFFSET[1], PHANTOM_OFFSET[2])
      m.rotation.order = 'YXZ'
      m.userData.fvPhantom = true
    }
    m.rotation.x = pitch || 0
  }

  window.fvGlInfo = function () {
    const gl = r.getContext()
    const dbg = gl.getExtension('WEBGL_debug_renderer_info')
    return { renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER), version: gl.getParameter(gl.VERSION) }
  }

  window.fvSetup = function (o) {
    // o: {zenith, horizon, fogNear, fogFar, sunDir:[x,y,z] (MC axes), showBot}
    const zen = new THREE.Color(o.zenith || '#4f7fe6')
    const hor = new THREE.Color(o.horizon || '#b5d0ff')
    v.scene.background = hor
    v.scene.fog = new THREE.Fog(hor, o.fogNear || 60, o.fogFar || 110)
    if (o.sunDir) sunDir = new THREE.Vector3(o.sunDir[0], o.sunDir[1], o.sunDir[2]).normalize()
    if (!sky) {
      const geo = new THREE.SphereGeometry(400, 32, 16)
      const mat = new THREE.ShaderMaterial({
        side: THREE.BackSide,
        depthWrite: false,
        uniforms: { top: { value: zen }, bottom: { value: hor }, sunDir: { value: sunDir } },
        vertexShader: 'varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
        // the lowest ~8 deg are exactly the fog colour, so fogged terrain at the chunk edge melts into the sky instead
        // of standing out as pale silhouettes; the zenith blend starts above that
        fragmentShader: 'uniform vec3 top; uniform vec3 bottom; uniform vec3 sunDir; varying vec3 vP; void main(){ float h = clamp(vP.y, 0.0, 1.0); vec3 c = mix(bottom, top, smoothstep(0.14, 0.8, h)); float g = max(dot(normalize(vP), sunDir), 0.0); c += vec3(1.0, 0.85, 0.6) * pow(g, 24.0) * 0.35; gl_FragColor = vec4(c, 1.0); }'
      })
      sky = new THREE.Mesh(geo, mat)
      sky.renderOrder = -10
      sky.frustumCulled = false
      v.scene.add(sky)
      const tex = new THREE.TextureLoader().load('textures/1.21.4/environment/sun.png')
      tex.magFilter = THREE.NearestFilter
      tex.minFilter = THREE.NearestFilter
      const sm = new THREE.MeshBasicMaterial({ map: tex, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, fog: false })
      sun = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), sm)
      sun.renderOrder = -9
      sun.frustumCulled = false
      v.scene.add(sun)
    } else {
      sky.material.uniforms.top.value = zen
      sky.material.uniforms.bottom.value = hor
      sky.material.uniforms.sunDir.value = sunDir
    }
    v.directionalLight.position.copy(sunDir)
    v.directionalLight.intensity = 0.55
    v.ambientLight.color = new THREE.Color(0xbbbbbb)
    if (!botMesh) {
      botMesh = new window.fvEntity('1.16.4', 'player', v.scene).mesh
      v.scene.add(botMesh)
    }
    return true
  }

  window.fvWait = async function (o) {
    // wait until the page holds every column Node sent and the mesher has been idle for `settleMs`
    const t0 = performance.now()
    const limit = (o && o.timeoutMs) || 3000
    const expect = (o && o.expectColumns) || 0
    const settle = (o && o.settleMs !== undefined) ? o.settleMs : 250
    let idleSince = null
    while (performance.now() - t0 < limit) {
      const cols = Object.keys(v.world.loadedChunks).length
      const idle = cols >= expect && v.world.sectionsOutstanding.size === 0
      if (idle) {
        if (idleSince === null) idleSince = performance.now()
        if (performance.now() - idleSince >= settle) break
      } else idleSince = null
      await new Promise((resolve) => setTimeout(resolve, 25))
    }
    return { outstanding: v.world.sectionsOutstanding.size, ms: performance.now() - t0, meshes: Object.keys(v.world.sectionMeshs).length, pageColumns: Object.keys(v.world.loadedChunks).length, entities: Object.keys(v.entities.entities).length }
  }

  window.fvRender = function (o) {
    v.update()
    const names = o.names || {}
    for (const id in v.entities.entities) if (names[id]) fixUv(v.entities.entities[id], names[id])
    const ents = o.ents || []
    for (const e of ents) {
      const m = v.entities.entities[e.id]
      if (!m) continue
      m.position.set(e.x, e.y, e.z)
      if (e.yaw !== undefined && e.yaw !== null) m.rotation.y = e.yaw
      if (e.name === 'phantom') placePhantom(m, e.pitch)
      fixUv(m, e.name)
      pose(m, e.name, e.walk || 0, e.speed || 0)
    }
    if (botMesh && o.bot) {
      botMesh.position.set(o.bot.x, o.bot.y, o.bot.z)
      botMesh.rotation.y = o.bot.yaw
      pose(botMesh, 'player', o.bot.walk || 0, o.bot.speed || 0)
    }
    const out = []
    for (const c of o.cams) {
      const w = c.w || o.w
      const h = c.h || o.h
      r.setPixelRatio(1)
      r.setSize(w, h, false)
      v.camera.fov = c.fov || 70
      v.camera.aspect = w / h
      v.camera.near = c.near || 0.05
      v.camera.far = 1000
      v.camera.updateProjectionMatrix()
      v.camera.position.set(c.pos[0], c.pos[1], c.pos[2])
      v.camera.up.set(0, 1, 0)
      v.camera.lookAt(c.look[0], c.look[1], c.look[2])
      if (sky) sky.position.copy(v.camera.position)
      if (sun) {
        sun.position.copy(v.camera.position).addScaledVector(sunDir, 350)
        sun.lookAt(v.camera.position)
      }
      if (botMesh) botMesh.visible = !c.firstPerson
      r.render(v.scene, v.camera)
      out.push(r.domElement.toDataURL('image/jpeg', o.q || 0.92))
    }
    return out
  }
  window.fvReady = true
})()
