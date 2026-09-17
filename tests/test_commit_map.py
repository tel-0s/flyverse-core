"""`resolve_commit` / `commit_equivalent`: reading a pre-rewrite commit id after a history rewrite.

    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/test_commit_map.py -q

The 2026-09-17 `git filter-repo` scrub renamed every commit in this repository. Frozen run records under ignored
`out/` directories are never edited (docs/INTERP.md 10.4 rule 30 (ii)), so the ids they carry no longer resolve and
every analyser that `git show`s a recorded commit has to read it through the rewrite's own map,
`docs/audits/commit_map_<date>.json`.

The cases below run against a throwaway git repository built in a temporary directory -- one commit, then the same
commit rewritten with `git commit --amend` so the first id is genuinely gone -- with its own `docs/audits/` map.
Nothing here touches this checkout's history or its map: the repository under test is an argument (`repo=`).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyverse.interp.common import commit_equivalent, commit_map, resolve_commit  # noqa: E402

FAKE_A = "a" * 39 + "1"                      # two map keys that share a prefix, and one id nothing maps
FAKE_B = "a" * 39 + "2"
FAKE_NEW = "b" * 40
FAKE_NEW_B = "c" * 40
UNKNOWN = "dead" + "0" * 36


def git(repo: Path, *args: str) -> str:
    """Run git in `repo` with no user, system or global configuration in scope (hermetic)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"))}
    env.update(GIT_CONFIG_GLOBAL=str(repo / "empty.gitconfig"), GIT_CONFIG_SYSTEM=str(repo / "empty.gitconfig"),
               GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid",
               GIT_AUTHOR_DATE="2026-01-01T00:00:00Z", GIT_COMMITTER_DATE="2026-01-01T00:00:00Z")
    done = subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, f"git {' '.join(args)}: {done.stderr.strip()}"
    return done.stdout.strip()


def write_map(repo: Path, name: str, table: dict[str, str]) -> None:
    audits = repo / "docs" / "audits"
    audits.mkdir(parents=True, exist_ok=True)
    (audits / name).write_text(json.dumps({"note": "test", "map": table}, indent=2) + "\n", encoding="utf-8")


class CommitMap(unittest.TestCase):
    """One rewritten commit, one map file, and the four answers `resolve_commit` can give."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="flyverse-commit-map-")
        cls.repo = Path(cls._tmp.name)
        (cls.repo / "empty.gitconfig").write_text("", encoding="utf-8")
        git(cls.repo, "init", "--quiet")
        (cls.repo / "flyverse").mkdir()
        (cls.repo / "flyverse" / "air.py").write_text("PLUME = 1\n", encoding="utf-8")
        git(cls.repo, "add", "flyverse/air.py")
        git(cls.repo, "commit", "--quiet", "-m", "before the rewrite")
        cls.old = git(cls.repo, "rev-parse", "HEAD")
        # The rewrite: same tree, new commit object -- exactly what filter-repo does to every commit.
        git(cls.repo, "commit", "--quiet", "--amend", "-m", "after the rewrite (identifiers scrubbed)")
        cls.new = git(cls.repo, "rev-parse", "HEAD")
        assert cls.old != cls.new
        # filter-repo leaves nothing reachable at the old id; an amend alone leaves it dangling, so prune it.
        git(cls.repo, "reflog", "expire", "--expire=now", "--expire-unreachable=now", "--all")
        git(cls.repo, "gc", "--prune=now", "--quiet")
        write_map(cls.repo, "commit_map_2026-01-02.json", {cls.old: cls.new, FAKE_A: FAKE_NEW, FAKE_B: FAKE_NEW_B})

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_an_id_that_still_resolves_is_returned_unchanged(self):
        self.assertEqual(resolve_commit(self.new, repo=self.repo), self.new)
        self.assertEqual(resolve_commit(self.new[:8], repo=self.repo), self.new[:8])

    def test_a_pre_rewrite_id_resolves_through_the_map(self):
        self.assertEqual(resolve_commit(self.old, repo=self.repo), self.new)

    def test_the_old_id_really_is_gone(self):
        """Without the map the read fails -- the defect rule 30 records, reproduced."""
        done = subprocess.run(["git", "show", f"{self.old}:flyverse/air.py"], cwd=self.repo, capture_output=True)
        self.assertNotEqual(done.returncode, 0)
        resolved = resolve_commit(self.old, repo=self.repo)
        shown = subprocess.run(["git", "show", f"{resolved}:flyverse/air.py"], cwd=self.repo, capture_output=True)
        self.assertEqual(shown.returncode, 0)
        self.assertEqual(shown.stdout.replace(b"\r\n", b"\n"), b"PLUME = 1\n")

    def test_a_unique_abbreviation_of_a_pre_rewrite_id_resolves(self):
        for n in (7, 8, 12, 40):
            self.assertEqual(resolve_commit(self.old[:n], repo=self.repo), self.new, n)

    def test_an_ambiguous_abbreviation_raises(self):
        with self.assertRaises(ValueError) as e:
            resolve_commit("a" * 10, repo=self.repo)
        self.assertIn("aaaaaaaaaa", str(e.exception))

    def test_an_unmapped_id_raises_naming_the_sha(self):
        with self.assertRaises(ValueError) as e:
            resolve_commit(UNKNOWN, repo=self.repo)
        self.assertIn(UNKNOWN, str(e.exception))

    def test_a_mapped_id_is_returned_even_when_the_new_commit_is_not_here(self):
        self.assertEqual(resolve_commit(FAKE_A, repo=self.repo), FAKE_NEW)

    def test_a_non_id_raises(self):
        for bad in ("", "HEAD", "not-a-sha", None):
            with self.assertRaises(ValueError):
                resolve_commit(bad, repo=self.repo)

    def test_commit_equivalent_across_the_rewrite_and_across_abbreviations(self):
        for a, b in ((self.old, self.new), (self.new, self.old), (self.old[:7], self.new),
                     (self.old, self.new[:7]), (self.new[:7], self.new), (self.old, self.old)):
            self.assertTrue(commit_equivalent(a, b, repo=self.repo), (a, b))

    def test_commit_equivalent_says_no_to_different_commits(self):
        for a, b in ((self.old, UNKNOWN), (self.new, UNKNOWN), (UNKNOWN, FAKE_A), (self.new, ""), ("", "")):
            self.assertFalse(commit_equivalent(a, b, repo=self.repo), (a, b))

    def test_maps_compose_in_date_order(self):
        """A second rewrite's map is read after the first's, so the oldest id still reaches the newest commit."""
        with tempfile.TemporaryDirectory(prefix="flyverse-commit-map-twice-") as name:
            repo = Path(name)
            (repo / "empty.gitconfig").write_text("", encoding="utf-8")
            git(repo, "init", "--quiet")
            (repo / "air.py").write_text("PLUME = 1\n", encoding="utf-8")
            git(repo, "add", "air.py")
            git(repo, "commit", "--quiet", "-m", "one")
            first = git(repo, "rev-parse", "HEAD")
            git(repo, "commit", "--quiet", "--amend", "-m", "two")
            second = git(repo, "rev-parse", "HEAD")
            git(repo, "commit", "--quiet", "--amend", "-m", "three")
            third = git(repo, "rev-parse", "HEAD")
            git(repo, "reflog", "expire", "--expire=now", "--expire-unreachable=now", "--all")
            git(repo, "gc", "--prune=now", "--quiet")
            self.assertEqual(len({first, second, third}), 3)
            write_map(repo, "commit_map_2026-01-02.json", {first: second})
            write_map(repo, "commit_map_2026-01-03.json", {second: third})
            self.assertEqual(resolve_commit(first, repo=repo), third)
            self.assertEqual(resolve_commit(second, repo=repo), third)
            self.assertTrue(commit_equivalent(first, second, repo=repo))

    def test_no_map_at_all_raises(self):
        with tempfile.TemporaryDirectory(prefix="flyverse-commit-map-bare-") as bare:
            self.assertEqual(commit_map(Path(bare)), {})
            with self.assertRaises(ValueError):
                resolve_commit(self.old, repo=Path(bare))


class ShippedMap(unittest.TestCase):
    """The map this repository ships is a table of full ids, and it answers for the frozen records that need it."""

    def test_every_entry_is_a_pair_of_full_ids(self):
        table = commit_map(ROOT)
        self.assertGreater(len(table), 100)
        for old, new in table.items():
            self.assertRegex(old, r"^[0-9a-f]{40}$")
            self.assertRegex(new, r"^[0-9a-f]{40}$")
        # filter-repo lists untouched commits too (old == new); the rewrite renamed most, not all.
        self.assertTrue(any(old != new for old, new in table.items()))

    def test_the_map_files_parse_and_declare_their_date(self):
        files = sorted((ROOT / "docs" / "audits").glob("commit_map_*.json"))
        self.assertTrue(files)
        for path in files:
            self.assertRegex(path.name, r"^commit_map_\d{4}-\d{2}-\d{2}\.json$")
            json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
