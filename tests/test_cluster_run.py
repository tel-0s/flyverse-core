"""Offline contracts for scripts/cluster_run.py: config parsing, tunnels, least-loaded scheduling, fetch guard.

Nothing here touches a real cluster: the heimdall API is a canned in-process HTTP server, ssh/scp are
patched out, and the tunnel subprocess is faked. No job is ever submitted anywhere.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import socket
import socketserver
import sys
import tempfile
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

_SPEC = importlib.util.spec_from_file_location(
    "cluster_run", Path(__file__).resolve().parents[1] / "scripts" / "cluster_run.py")
cr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cr)


TARGET = {"ssh": "root@1.2.3.4", "api": "http://127.0.0.1:7000/api/v1",
          "root": "/root/flyverse", "runs": "/root/runs", "user": "tel0s"}


def target(name="a", **over):
    return cr.Target(name, {**TARGET, **over})


# ── a canned heimdall ────────────────────────────────────────────────────────

NODES = {"node1": {"node": "node1", "reachable": True, "fence_reason": None,
                   "gpus": [{"index": 0, "memory_total_mb": 183 * 1024},
                            {"index": 1, "memory_total_mb": 183 * 1024}]}}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):                                          # keep the test output clean
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        st = self.server.state
        u = urllib.parse.urlsplit(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path.endswith("/status"):
            return self._send({"nodes": list(NODES), "jobs": []})
        if u.path.endswith("/nodes"):
            return self._send(st.nodes)
        if u.path.endswith("/api/v1/jobs"):
            want = (q.get("status") or [None])[0]
            jobs = [j for j in st.listing() if want is None or j["status"] == want]
            return self._send({"jobs": jobs, "total": len(jobs)})
        m = re.match(r".*/jobs/([^/]+)/logs$", u.path)
        if m:
            return self._send({"job_id": m.group(1), "log": "ALSA lib pcm.c noise\nhello from the job\n"})
        m = re.match(r".*/jobs/([^/]+)$", u.path)
        if m:
            return self._send(st.poll(m.group(1)))
        return self._send({"detail": "not found"}, 404)

    def do_POST(self):
        st = self.server.state
        n = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path.endswith("/api/v1/jobs"):
            return self._send(st.submit(body))
        return self._send({"detail": "not found"}, 404)


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):                                              # skip getfqdn(): a 1 s stall per server here
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


class FakeHeimdall:
    """POST /jobs, GET /jobs, /jobs/{id}, /jobs/{id}/logs, /nodes, /status -- enough for cluster_run."""

    def __init__(self, existing=(), nodes=None, fail=False):
        self.existing = [dict(j) for j in existing]
        self.nodes = nodes if nodes is not None else NODES
        self.fail = fail
        self.submitted = []                                             # request bodies, in order
        self.jobs = {}
        self.polls = {}
        self.server = _Server(("127.0.0.1", 0), _Handler)
        self.server.state = self
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    @property
    def api(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}/api/v1"

    def listing(self):
        return self.existing + [dict(j, status="queued") for j in self.jobs.values()]

    def submit(self, body):
        jid = f"job{len(self.jobs) + 1:02d}"
        self.submitted.append(body)
        self.jobs[jid] = {"id": jid, "status": "queued", "submitted_by": body.get("submitted_by"),
                          "name": body["spec"]["name"], "exit_code": None}
        return {"job": dict(self.jobs[jid]), "warnings": []}

    def poll(self, jid):
        self.polls[jid] = self.polls.get(jid, 0) + 1
        j = dict(self.jobs[jid])
        if self.polls[jid] <= 1:
            j["status"] = "running"
        else:
            j["status"] = "failed" if self.fail else "completed"
            j["exit_code"] = 1 if self.fail else 0
        return j

    def names(self):
        return [b["spec"]["name"] for b in self.submitted]

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def run_main(argv, config, **patches):
    """Run cluster_run.main() with ssh/scp/git patched out; returns (exit code, console log)."""
    buf = io.StringIO()
    env = dict(os.environ, FLYVERSE_CLUSTER=json.dumps(config))
    calls = {"ssh": [], "scp": []}

    def fake_ssh(t, cmd, check=True, input_bytes=None, retry=True):
        calls["ssh"].append((t.name, cmd))
        return ""

    def fake_scp(t, src, dst, retry=True):
        calls["scp"].append((t.name, src, dst))
        return 0

    with patch.dict(os.environ, env, clear=False), patch.object(sys, "argv", ["cluster_run.py", *argv]), \
            patch.object(cr, "ship_list", lambda: []), patch.object(cr, "ssh", fake_ssh), \
            patch.object(cr, "scp", fake_scp), patch.object(cr, "PROBE_TIMEOUT", 1.0), \
            patch.object(cr, "RETRY_DELAY", 0.0), contextlib.redirect_stdout(buf):
        try:
            code = cr.main()
        except SystemExit as e:
            code = e.code
    return code, buf.getvalue(), calls


# ── config ───────────────────────────────────────────────────────────────────

class ConfigTests(unittest.TestCase):
    def test_single_object_is_one_target_with_the_old_defaults(self):
        targets, default = cr.parse_config(dict(TARGET, env={"PATH": "/bin"}, gpus=1, vram_gb=24))
        self.assertEqual(list(targets), [cr.SINGLE])
        self.assertEqual(default, [cr.SINGLE])
        t = targets[cr.SINGLE]
        self.assertEqual((t.host, t.port, t.tunnel, t.slots, t.disabled), ("root@1.2.3.4", 22, False, None, False))
        self.assertEqual((t.root, t.runs, t.user, t.gpus, t.vram_gb), ("/root/flyverse", "/root/runs", "tel0s", 1, 24))
        self.assertEqual(t.api, t.remote_api)                           # no tunnel: the API url is used as written

    def test_single_object_missing_a_required_key_exits(self):
        with self.assertRaises(SystemExit) as e:
            cr.parse_config({k: v for k, v in TARGET.items() if k != "runs"})
        self.assertIn("runs", str(e.exception))

    def test_multi_target_config_inherits_shared_keys_and_merges_env(self):
        cfg = {"user": "tel0s", "env": {"PATH": "/bin", "CUDA_HOME": "/usr/local/cuda"},
               "targets": {"house": dict(TARGET, ssh="athuser@node1", user="tel0s"),
                           "box-a": dict(TARGET, port=41234, tunnel=True, slots=4,
                                         env={"PATH": "/usr/bin"})},
               "default": ["house"]}
        targets, default = cr.parse_config(cfg)
        self.assertEqual(sorted(targets), ["box-a", "house"])
        self.assertEqual(default, ["house"])
        box = targets["box-a"]
        self.assertEqual((box.port, box.tunnel, box.slots), (41234, True, 4))
        self.assertEqual(box.env, {"PATH": "/usr/bin", "CUDA_HOME": "/usr/local/cuda"})
        self.assertEqual(targets["house"].port, 22)                     # ssh port defaults to 22
        self.assertEqual(targets["house"].user, "tel0s")

    def test_default_list_absent_means_every_enabled_target(self):
        cfg = {"targets": {"a": dict(TARGET), "b": dict(TARGET, disabled=True), "c": dict(TARGET)}}
        targets, default = cr.parse_config(cfg)
        self.assertEqual(default, ["a", "c"])
        self.assertEqual([t.name for t in cr.select_targets(targets, default, None)], ["a", "c"])

    def test_disabled_target_named_in_the_default_list_is_skipped(self):
        cfg = {"targets": {"a": dict(TARGET), "b": dict(TARGET, disabled=True)}, "default": ["a", "b"]}
        targets, default = cr.parse_config(cfg)
        self.assertEqual([t.name for t in cr.select_targets(targets, default, None)], ["a"])

    def test_explicitly_selecting_a_disabled_or_unknown_target_exits(self):
        targets, default = cr.parse_config({"targets": {"a": dict(TARGET), "b": dict(TARGET, disabled=True)}})
        with self.assertRaises(SystemExit) as e:
            cr.select_targets(targets, default, ["b"])
        self.assertIn("disabled", str(e.exception))
        with self.assertRaises(SystemExit) as e:
            cr.select_targets(targets, default, ["zzz"])
        self.assertIn("unknown target", str(e.exception))

    def test_selection_keeps_the_flag_order_and_dedupes(self):
        targets, default = cr.parse_config({"targets": {"a": dict(TARGET), "b": dict(TARGET), "c": dict(TARGET)},
                                            "default": ["a"]})
        self.assertEqual([t.name for t in cr.select_targets(targets, default, ["c", "a", "c"])], ["c", "a"])


# ── ssh, scp, tunnels ────────────────────────────────────────────────────────

class SshTests(unittest.TestCase):
    def test_ssh_and_scp_pass_the_port(self):
        t = target(port=41234)
        seen = []

        class R:
            returncode = 0
            stdout = b""
            stderr = b""

        def fake_run(argv, **kw):
            seen.append(argv)
            return R()

        with patch.object(cr.subprocess, "run", fake_run):
            cr.ssh(t, "echo hi")
            cr.scp(t, "root@1.2.3.4:/root/runs/x/out/a.json", "/local/a.json")
        self.assertEqual(seen[0][:2], ["ssh", "-o"])
        self.assertIn("-p", seen[0])
        self.assertEqual(seen[0][seen[0].index("-p") + 1], "41234")
        self.assertEqual(seen[0][-2:], ["root@1.2.3.4", "echo hi"])
        self.assertIn("-P", seen[1])
        self.assertEqual(seen[1][seen[1].index("-P") + 1], "41234")

    def test_ssh_failure_raises_instead_of_killing_the_run(self):
        class R:
            returncode = 255
            stdout = b""
            stderr = b"Connection refused"

        with patch.object(cr.subprocess, "run", lambda *a, **k: R()), patch.object(cr, "RETRY_DELAY", 0.0):
            with self.assertRaises(cr.ClusterError):
                cr.ssh(target(), "true")


class TunnelTests(unittest.TestCase):
    def test_free_local_port_is_free(self):
        p = cr.free_local_port()
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", p))                                    # would raise if it were taken
        finally:
            s.close()

    def test_url_port_and_local_rewrite(self):
        self.assertEqual(cr.url_port("http://127.0.0.1:7000/api/v1"), 7000)
        self.assertEqual(cr.url_port("http://host/api/v1"), 80)
        self.assertEqual(cr.url_port("https://host/api/v1"), 443)
        self.assertEqual(cr.local_api_url("http://127.0.0.1:7000/api/v1", 55001), "http://127.0.0.1:55001/api/v1")

    def test_tunnel_command_forwards_the_remote_api_port_over_the_ssh_port(self):
        t = target(port=41234, tunnel=True)
        cmd = cr.tunnel_cmd(t, 55001)
        self.assertEqual(cmd[0], "ssh")
        for flag in ("-N", "BatchMode=yes", "ExitOnForwardFailure=yes"):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index("-L") + 1], "55001:127.0.0.1:7000")
        self.assertEqual(cmd[cmd.index("-p") + 1], "41234")
        self.assertEqual(cmd[-1], "root@1.2.3.4")

    def test_open_tunnel_rewrites_the_api_url_and_close_kills_it(self):
        t = target(port=41234, tunnel=True)
        killed = []

        class P:
            returncode = None
            stderr = None

            def poll(self):
                return None

            def terminate(self):
                killed.append(True)

            def wait(self, timeout=None):
                return 0

        with patch.object(cr.subprocess, "Popen", lambda *a, **k: P()), contextlib.redirect_stdout(io.StringIO()):
            url = cr.open_tunnel(t)
        self.assertTrue(url.startswith("http://127.0.0.1:"))
        self.assertTrue(url.endswith("/api/v1"))
        self.assertNotEqual(url, t.remote_api)
        self.assertEqual(t.api, url)
        self.assertIn(t, cr._TUNNELS)
        cr.close_tunnels()
        self.assertEqual(killed, [True])
        self.assertEqual(cr._TUNNELS, [])


# ── capacity and assignment ──────────────────────────────────────────────────

class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.h = FakeHeimdall(existing=[
            {"id": "old1", "status": "running", "submitted_by": "tel0s", "name": "x", "exit_code": None},
            {"id": "old2", "status": "queued", "submitted_by": "tel0s", "name": "y", "exit_code": None},
            {"id": "old3", "status": "running", "submitted_by": "someone-else", "name": "z", "exit_code": None},
            {"id": "old4", "status": "completed", "submitted_by": "tel0s", "name": "w", "exit_code": 0},
        ])
        self.addCleanup(self.h.stop)

    def test_slots_estimated_from_the_node_list(self):
        t = target(api=self.h.api, vram_gb=24)
        self.assertEqual(cr.estimate_slots(t), 7 * 2)                   # floor((183 - 4) / 24) = 7, two GPUs
        self.assertEqual(cr.estimate_slots(target(api=self.h.api, vram_gb=90)), 1 * 2)

    def test_slots_estimate_ignores_unreachable_and_fenced_nodes(self):
        h = FakeHeimdall(nodes={"n1": {"reachable": False, "gpus": [{"memory_total_mb": 183 * 1024}]},
                                "n2": {"reachable": True, "fence_reason": "gpu recovery",
                                       "gpus": [{"memory_total_mb": 183 * 1024}]}})
        self.addCleanup(h.stop)
        with self.assertRaises(cr.ClusterError):
            cr.estimate_slots(target(api=h.api))

    def test_current_load_counts_only_our_pending_queued_running_jobs(self):
        self.assertEqual(cr.current_load(target(api=self.h.api, user="tel0s")), 2)
        self.assertEqual(cr.current_load(target(api=self.h.api, user="nobody")), 0)

    def test_pick_target_is_least_loaded_first_and_stable_on_ties(self):
        a, b = target("a", slots=2), target("b", slots=4)
        b.load = 3                                                      # a: 2 free, b: 1 free
        picks = []
        for _ in range(5):
            t = cr.pick_target([a, b])
            t.load += 1
            picks.append(t.name)
        self.assertEqual(picks, ["a", "a", "b", "a", "b"])              # ties go to the earlier target
        c = target("c", slots=1)
        c.ok = False
        self.assertEqual(cr.pick_target([c, a]).name, "a")              # an unavailable target is never picked


# ── the run itself ───────────────────────────────────────────────────────────

class RunTests(unittest.TestCase):
    def two_boxes(self, slots=(1, 3), **kw):
        a, b = FakeHeimdall(**kw), FakeHeimdall(**kw)
        self.addCleanup(a.stop)
        self.addCleanup(b.stop)
        cfg = {"user": "tel0s", "targets": {
            "a": dict(TARGET, api=a.api, slots=slots[0], runs="/root/runs-a"),
            "b": dict(TARGET, api=b.api, slots=slots[1], runs="/root/runs-b")}}
        return a, b, cfg

    def test_every_job_line_records_its_target_and_the_log_ends_with_the_totals(self):
        a, b, cfg = self.two_boxes(slots=(1, 1))
        code, log, calls = run_main(["--name", "x", "--poll", "0.01", "c0", "c1"], cfg)
        self.assertEqual(code, 0)
        job_lines = [l for l in log.splitlines() if l.startswith("job ")]
        self.assertEqual(len(job_lines), 2)
        self.assertTrue(all(re.search(r"@(a|b)\s", l) for l in job_lines), job_lines)
        self.assertIn("@a", log)
        self.assertIn("@b", log)
        # each box hands out its own job ids, so the two jobs here share the id "job01": the run must keep
        # them apart by (target, id), not by id alone.
        self.assertEqual([b_["spec"]["name"].rsplit("-", 1)[-1] for b_ in a.submitted], ["0"])
        self.assertEqual([b_["spec"]["name"].rsplit("-", 1)[-1] for b_ in b.submitted], ["1"])
        self.assertRegex(log, r"---- x-\w+-0 @a \(job\d+\) completed exit 0: c0 ----")
        self.assertRegex(log, r"---- x-\w+-1 @b \(job\d+\) completed exit 0: c1 ----")
        self.assertIn("  @a: 1 job(s), 0 failed  run dir /root/runs-a/x-", log)
        self.assertIn("  @b: 1 job(s), 0 failed  run dir /root/runs-b/x-", log)
        last = [l for l in log.splitlines() if l.strip()][-1]
        self.assertRegex(last, r"^2 job\(s\), 0 failed  \(\d+\.\d min\)  run dir x-\w+ on 2 target\(s\)$")
        self.assertNotIn("ALSA lib", log)                               # the ALSA filter still applies
        self.assertEqual(sorted(t for t, _ in calls["ssh"]), ["a", "b"])  # the diff went to both boxes

    def test_commands_go_to_the_least_loaded_target(self):
        a, b, cfg = self.two_boxes()                                    # a has 1 slot, b has 3
        code, log, _ = run_main(["--name", "x", "--poll", "0.01", "c0", "c1", "c2", "c3"], cfg)
        self.assertEqual(code, 0)
        self.assertEqual(len(a.submitted), 1)
        self.assertEqual(len(b.submitted), 3)
        # b has 3 free to a's 1, so it takes c0 and c1; at 1 free each the tie goes to a (c2); then b again.
        self.assertEqual([n.rsplit("-", 1)[-1] for n in a.names()], ["2"])
        self.assertEqual([n.rsplit("-", 1)[-1] for n in b.names()], ["0", "1", "3"])
        spec = b.submitted[0]["spec"]
        self.assertEqual(spec["command"], "source .venv/bin/activate && c0")
        self.assertTrue(spec["working_dir"].startswith("/root/runs-b/x-"))
        self.assertEqual(spec["log_path"], f"{spec['working_dir']}/logs/{spec['name']}.log")
        self.assertEqual((spec["gpus"], spec["vram_gb"]), (1, 24))
        self.assertEqual(spec["env"]["TORCH_EXTENSIONS_DIR"], "/root/flyverse/.torch_ext")
        self.assertEqual(b.submitted[0]["submitted_by"], "tel0s")

    def test_a_targets_existing_jobs_count_against_its_free_slots(self):
        mine = [{"id": f"o{i}", "status": "running", "submitted_by": "tel0s", "name": "old", "exit_code": None}
                for i in range(3)]
        a = FakeHeimdall()
        b = FakeHeimdall(existing=mine)                                 # b: 3 slots, 3 of ours already on it
        self.addCleanup(a.stop)
        self.addCleanup(b.stop)
        cfg = {"user": "tel0s", "targets": {"a": dict(TARGET, api=a.api, slots=1),
                                            "b": dict(TARGET, api=b.api, slots=3)}}
        code, log, _ = run_main(["--name", "x", "--poll", "0.01", "c0"], cfg)
        self.assertEqual(code, 0)
        self.assertIn("3 job(s) of tel0s already pending/queued/running", log)
        self.assertEqual(len(a.submitted), 1)
        self.assertEqual(len(b.submitted), 0)

    def test_a_silent_target_is_dropped_and_the_others_carry_the_batch(self):
        a, b, cfg = self.two_boxes()
        b.stop()                                                        # b's API stops answering before we start
        code, log, _ = run_main(["--name", "x", "--poll", "0.01", "c0", "c1"], cfg)
        self.assertEqual(code, 0)
        self.assertIn("[b] UNAVAILABLE", log)
        self.assertEqual(len(a.submitted), 2)
        self.assertRegex([l for l in log.splitlines() if l.strip()][-1], r"^2 job\(s\), 0 failed")

    def test_every_target_silent_exits_non_zero_without_submitting(self):
        a, b, cfg = self.two_boxes()
        a.stop()
        b.stop()
        code, log, calls = run_main(["--name", "x", "--poll", "0.01", "c0"], cfg)
        self.assertEqual(code, 2)
        self.assertIn("no target answered", log)
        self.assertEqual(calls["ssh"], [])

    def test_a_failed_job_is_counted_per_target_and_exits_one(self):
        a, b, cfg = self.two_boxes(slots=(1, 1), fail=True)
        code, log, _ = run_main(["--name", "x", "--poll", "0.01", "c0", "c1"], cfg)
        self.assertEqual(code, 1)
        self.assertIn("  @a: 1 job(s), 1 failed", log)
        self.assertIn("  @b: 1 job(s), 1 failed", log)
        self.assertRegex([l for l in log.splitlines() if l.strip()][-1], r"^2 job\(s\), 2 failed")

    def test_single_target_config_still_runs_and_reports_its_run_dir(self):
        h = FakeHeimdall()
        self.addCleanup(h.stop)
        cfg = dict(TARGET, api=h.api, slots=2)
        code, log, calls = run_main(["--name", "x", "--poll", "0.01", "c0"], cfg)
        self.assertEqual(code, 0)
        self.assertEqual(len(h.submitted), 1)
        self.assertRegex(h.names()[0], r"^x-\w{6}$")                    # one command: no -<i> suffix, as before
        self.assertIn(f"@{cr.SINGLE}", log)
        self.assertRegex([l for l in log.splitlines() if l.strip()][-1],
                         r"^1 job\(s\), 0 failed  \(\d+\.\d min\)  run dir /root/runs/x-\w+$")
        self.assertEqual([t for t, _ in calls["ssh"]], [cr.SINGLE])

    def test_sync_only_ships_everywhere_and_submits_nothing(self):
        a, b, cfg = self.two_boxes()
        code, log, calls = run_main(["--name", "x", "--sync-only", "c0"], cfg)
        self.assertEqual(code, 0)
        self.assertEqual(sorted(t for t, _ in calls["ssh"]), ["a", "b"])
        self.assertEqual((a.submitted, b.submitted), ([], []))

    def test_targets_flag_restricts_the_run_to_one_box(self):
        a, b, cfg = self.two_boxes()
        code, log, calls = run_main(["--name", "x", "--target", "b", "--poll", "0.01", "c0", "c1"], cfg)
        self.assertEqual(code, 0)
        self.assertEqual(len(b.submitted), 2)
        self.assertEqual(a.submitted, [])
        self.assertEqual(sorted({t for t, _ in calls["ssh"]}), ["b"])


# ── fetch ────────────────────────────────────────────────────────────────────

class FetchTests(unittest.TestCase):
    def test_bare_fetch_paths_are_recognised(self):
        self.assertEqual(cr.bare_fetch_paths(["out/"]), ["out/"])
        self.assertEqual(cr.bare_fetch_paths(["out", "out/", "."]), ["out", "out/", "."])
        self.assertEqual(cr.bare_fetch_paths(["out/run/", "out/x.json", "logs/"]), [])

    def test_bare_fetch_is_refused_with_the_reason(self):
        h = FakeHeimdall()
        self.addCleanup(h.stop)
        code, log, calls = run_main(["--name", "x", "c0", "--fetch", "out/"], dict(TARGET, api=h.api))
        self.assertIsInstance(code, str)                                # sys.exit(message)
        self.assertIn("--fetch out/", code)
        self.assertIn("cache_*", code)
        self.assertIn("--allow-bare-fetch", code)
        self.assertEqual(h.submitted, [])                               # refused before anything was shipped
        self.assertEqual(calls["ssh"], [])

    def test_allow_bare_fetch_lets_it_through(self):
        h = FakeHeimdall()
        self.addCleanup(h.stop)
        with patch.object(cr, "fetch_all", lambda targets, paths: None):
            code, log, _ = run_main(["--name", "x", "--poll", "0.01", "--allow-bare-fetch", "c0", "--fetch", "out/"],
                                    dict(TARGET, api=h.api, slots=1))
        self.assertEqual(code, 0)
        self.assertEqual(len(h.submitted), 1)

    def test_fetch_merges_both_targets_into_one_local_path_one_at_a_time(self):
        a, b = target("a"), target("b")
        for t, n in ((a, 1), (b, 2)):
            t.rdir = f"/root/runs-{t.name}/x-abc"
            t.njobs = n
        idle = target("c")                                              # ran nothing: never fetched from
        seen, overlap = [], []
        lock = threading.Lock()
        busy = {"n": 0}

        def fake_scp(t, src, dst, retry=True):
            with lock:
                busy["n"] += 1
                if busy["n"] > 1:
                    overlap.append(t.name)
            seen.append((t.name, src, dst))
            with lock:
                busy["n"] -= 1
            return 0

        with tempfile.TemporaryDirectory() as d, patch.object(cr, "scp", fake_scp), \
                patch.object(cr, "ROOT", d), contextlib.redirect_stdout(io.StringIO()) as buf:
            cr.fetch_all([a, b, idle], ["out/run/"])
        self.assertEqual([t for t, _, _ in seen], ["a", "b"])
        self.assertEqual(overlap, [])
        self.assertEqual(seen[0][1], "root@1.2.3.4:/root/runs-a/x-abc/out/run/.")   # dir/. merges the contents
        self.assertIn("fetched out/run/ @a", buf.getvalue())
        self.assertIn("fetched out/run/ @b", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
