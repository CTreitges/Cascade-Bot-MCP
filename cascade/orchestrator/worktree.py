"""WorktreeManager — git-worktree-basierte Isolation pro Sub-Task.

Jeder parallele Sub-Task bekommt seinen eigenen working directory via
`git worktree add`. Vorteile gegenüber File-Locking oder vollem Klon:
  - native git-Konzept, sauber + reversibel
  - schnell (shared object store, copy-on-write)
  - kein Konflikt zwischen parallelen Schreibern
  - jedem Sub-Task eigener Branch → Reviewer kann später diffen

Lifecycle:
  create(sub_task_id)          erzeugt Worktree + Branch
  remove(sub_task_id)          räumt Worktree weg, Branch bleibt
  cleanup_all()                Notfall-Cleanup (z.B. beim Bot-Shutdown)
  list_active()                aktuelle Worktrees

Naming:
  Worktree-Pfad:  <repo>/.cascade-worktrees/<sub-task-id>/
  Branch-Name:    cascade/sub-<sub-task-id>

Beim Bot-Restart kann lifecycle.post_init `cleanup_all()` aufrufen, um
hängengebliebene Worktrees zu entfernen.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


logger = logging.getLogger("cascade.orchestrator.worktree")


_BRANCH_PREFIX = "cascade/sub-"
_WORKTREE_DIR = ".cascade-worktrees"


def _safe_id(s: str) -> str:
    """Sanitize Sub-Task-Name → Filesystem/Branch-safe."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", s).strip("-").lower() or "subtask"


@dataclass
class Worktree:
    sub_task_id: str
    path: Path
    branch: str

    @property
    def cwd(self) -> str:
        return str(self.path)


class WorktreeManager:
    """Verwaltet git-worktrees für einen Cascade-Run.

    Args:
        repo_root: Root des Source-Repos (muss git init'd sein).
        base_ref: Branch oder Commit von dem Worktrees abzweigen
                  (default: aktueller HEAD).
    """

    def __init__(self, repo_root: Path | str, base_ref: Optional[str] = None) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.base_ref = base_ref or "HEAD"
        self.workdir = self.repo_root / _WORKTREE_DIR
        self.active: Dict[str, Worktree] = {}

    # ── Internal: subprocess wrapper ───────────────────────────────────
    async def _run_git(
        self, *args: str, cwd: Optional[Path] = None, check: bool = False
    ) -> Tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            cwd=str(cwd or self.repo_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_EDITOR": "true"},
        )
        stdout, stderr = await proc.communicate()
        rc = proc.returncode or 0
        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()
        if check and rc != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed (rc={rc}): {err[:200]}"
            )
        return rc, out, err

    # ── Public API ─────────────────────────────────────────────────────
    async def create(self, sub_task_id: str) -> Worktree:
        """Erzeugt Worktree + neuer Branch für diesen Sub-Task.

        Wenn Worktree-Pfad oder Branch schon existieren (z.B. von einem
        gecrashten vorherigen Run), werden sie zuerst aufgeräumt.
        """
        sid = _safe_id(sub_task_id)
        path = self.workdir / sid
        branch = f"{_BRANCH_PREFIX}{sid}"

        # Existing-Cleanup defensiv
        if path.exists():
            logger.warning("worktree path exists, removing: %s", path)
            await self._run_git("worktree", "remove", "--force", str(path))
        # Branch evtl. noch da → löschen (wir wollen einen frischen)
        rc, _, _ = await self._run_git("rev-parse", "--verify", "--quiet", f"refs/heads/{branch}")
        if rc == 0:
            await self._run_git("branch", "-D", branch)

        self.workdir.mkdir(parents=True, exist_ok=True)
        await self._run_git(
            "worktree", "add", "-b", branch, str(path), self.base_ref,
            check=True,
        )

        wt = Worktree(sub_task_id=sub_task_id, path=path, branch=branch)
        self.active[sub_task_id] = wt
        logger.info("worktree created: %s @ %s (branch=%s)", sub_task_id, path, branch)
        return wt

    async def commit_changes(
        self,
        sub_task_id: str,
        message: str,
    ) -> bool:
        """Committed alle Änderungen im Worktree. Returns True wenn was gecommittet
        wurde, False wenn der Worktree clean war."""
        wt = self.active.get(sub_task_id)
        if not wt:
            return False
        # Status-Porcelain checken
        rc, out, _ = await self._run_git("status", "--porcelain", cwd=wt.path)
        if rc != 0 or not out.strip():
            return False
        await self._run_git("add", "-A", cwd=wt.path)
        rc, _, err = await self._run_git(
            "-c", "user.email=cascade@local",
            "-c", "user.name=Cascade-Subtask",
            "commit", "-m", message,
            cwd=wt.path,
        )
        return rc == 0

    async def get_diff_against_base(self, sub_task_id: str) -> str:
        """Diff zwischen base_ref und dem Sub-Task-Branch."""
        wt = self.active.get(sub_task_id)
        if not wt:
            return ""
        rc, out, _ = await self._run_git(
            "diff", f"{self.base_ref}..{wt.branch}",
        )
        return out if rc == 0 else ""

    async def get_files_changed(self, sub_task_id: str) -> List[str]:
        """Liste der geänderten Files zwischen base_ref und Sub-Task-Branch."""
        wt = self.active.get(sub_task_id)
        if not wt:
            return []
        rc, out, _ = await self._run_git(
            "diff", "--name-only", f"{self.base_ref}..{wt.branch}",
        )
        if rc != 0:
            return []
        return [line for line in out.splitlines() if line.strip()]

    async def remove(self, sub_task_id: str, keep_branch: bool = True) -> None:
        """Räumt Worktree weg. Branch bleibt per Default — für Merge-Phase."""
        wt = self.active.pop(sub_task_id, None)
        if not wt:
            return
        await self._run_git("worktree", "remove", "--force", str(wt.path))
        if not keep_branch:
            await self._run_git("branch", "-D", wt.branch)

    async def cleanup_all(self) -> None:
        """Notfall-Cleanup: räumt alle .cascade-worktrees/* + cascade/sub-*-Branches.

        Wird von lifecycle.post_init aufgerufen um Reste eines vorigen Crashes
        zu entfernen.
        """
        if self.workdir.exists():
            for child in list(self.workdir.iterdir()):
                if child.is_dir():
                    await self._run_git(
                        "worktree", "remove", "--force", str(child)
                    )
            # Pruning falls remove failed
            await self._run_git("worktree", "prune")
        # Verwaiste cascade/sub-*-Branches
        rc, out, _ = await self._run_git(
            "branch", "--list", f"{_BRANCH_PREFIX}*",
        )
        if rc == 0:
            for line in out.splitlines():
                name = line.strip().lstrip("*").strip()
                if name.startswith(_BRANCH_PREFIX):
                    await self._run_git("branch", "-D", name)
        self.active.clear()

    def list_active(self) -> List[Worktree]:
        return list(self.active.values())
