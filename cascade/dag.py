"""DAG-Helfer für Plan v4 Sub-Task-Orchestrierung.

Bietet:
  - validate_dag(plan)           Strukturprüfung (Zyklus, dangling refs, Name-
                                 Eindeutigkeit). Returnt list[str] der
                                 menschenlesbaren Fehler, leer = OK.
  - topological_batches(plan)    Liefert Liste von Batches, jede Batch enthält
                                 die Sub-Tasks die in dieser Phase parallel
                                 laufen können (Kahn's Algorithmus + File-
                                 Overlap-Aufteilung).
  - has_file_overlap(subtasks)   Prüft Files-To-Touch-Disjunktion innerhalb
                                 einer Batch.

Wird von core.py NACH Plan-Validierung aufgerufen. Wenn validate_dag Fehler
zurückgibt, sollte der Caller einen Replan triggern.

Phase D: Modul + Validation aktiv genutzt.
Phase E: topological_batches() wird vom Orchestrator konsumiert.
"""
from __future__ import annotations

import fnmatch
from typing import Iterable, List, Sequence


# ──────────────────────────────────────────────────────────────────────────────
#  Validation
# ──────────────────────────────────────────────────────────────────────────────
def validate_dag(subtasks: Sequence) -> List[str]:
    """Prüft Plan auf DAG-Konsistenz. `subtasks` ist eine Sequence von SubTask
    (jedes Element muss .name + .depends_on haben)."""
    errors: List[str] = []
    if not subtasks:
        return errors

    names = [st.name for st in subtasks]
    name_set = set(names)

    # Eindeutige Namen?
    seen = set()
    for n in names:
        if n in seen:
            errors.append(f"sub-task name '{n}' kommt mehrfach vor")
        seen.add(n)

    # Dangling refs?
    for st in subtasks:
        for dep in st.depends_on:
            if dep not in name_set:
                errors.append(
                    f"sub-task '{st.name}' verweist via depends_on auf "
                    f"unbekanntes '{dep}'"
                )

    # Self-loop?
    for st in subtasks:
        if st.name in st.depends_on:
            errors.append(f"sub-task '{st.name}' hängt von sich selbst ab")

    # Zykel-Check (DFS-basiert)
    if not errors:
        cycle = _find_cycle(subtasks)
        if cycle:
            errors.append(
                "DAG enthält einen Zykel: " + " → ".join(cycle + [cycle[0]])
            )

    return errors


def _find_cycle(subtasks: Sequence) -> List[str] | None:
    """Findet den ersten Zykel via DFS. Liefert die Knotenliste oder None."""
    by_name = {st.name: st for st in subtasks}
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in by_name}
    parent: dict[str, str | None] = {n: None for n in by_name}

    def dfs(node: str) -> List[str] | None:
        color[node] = GRAY
        for dep in by_name[node].depends_on:
            if color.get(dep) == GRAY:
                # Zykel zwischen `node` und `dep`. Pfad rekonstruieren.
                cycle = [node]
                cur = parent[node]
                while cur is not None and cur != dep:
                    cycle.append(cur)
                    cur = parent[cur]
                cycle.append(dep)
                return list(reversed(cycle))
            if color.get(dep) == WHITE:
                parent[dep] = node
                found = dfs(dep)
                if found:
                    return found
        color[node] = BLACK
        return None

    for n in by_name:
        if color[n] == WHITE:
            res = dfs(n)
            if res:
                return res
    return None


# ──────────────────────────────────────────────────────────────────────────────
#  File-Overlap
# ──────────────────────────────────────────────────────────────────────────────
def _expand_globs(patterns: Iterable[str]) -> List[str]:
    """Globs werden NICHT gegen Dateisystem expandiert — nur als Pattern-Strings
    behandelt. Wir vergleichen sie als Strings."""
    return [p.strip() for p in patterns if p and p.strip()]


def _patterns_overlap(a: str, b: str) -> bool:
    """Prüft grob ob zwei Glob-Patterns überlappen können.

    Heuristik:
      - identische Strings → overlap
      - einer ist Suffix des anderen → overlap
      - fnmatch in beide Richtungen → wenn match, overlap
      - common prefix mit '**' am Ende → overlap angenommen
    """
    if a == b:
        return True
    if fnmatch.fnmatch(a, b) or fnmatch.fnmatch(b, a):
        return True
    # Beide könnten Globs sein → Pattern-vs-Pattern grob über fnmatch reicht nicht;
    # konservativ: gleicher Verzeichnis-Prefix → annehmen overlap.
    pref = _common_dir_prefix(a, b)
    if pref and ("**" in a or "**" in b or "*" in a or "*" in b):
        return True
    return False


def _common_dir_prefix(a: str, b: str) -> str:
    pa = a.split("/")
    pb = b.split("/")
    common: List[str] = []
    for x, y in zip(pa, pb):
        if x == y and "*" not in x:
            common.append(x)
        else:
            break
    return "/".join(common)


def has_file_overlap(subtasks: Sequence) -> List[tuple[str, str, str, str]]:
    """Gibt eine Liste von Konflikten innerhalb der gegebenen Sub-Tasks zurück.

    Jeder Eintrag: (sub_task_a, pattern_a, sub_task_b, pattern_b).
    Leere Liste = keine Überlappung, Sub-Tasks sind sicher parallelisierbar.
    """
    conflicts: List[tuple[str, str, str, str]] = []
    items = [(st.name, _expand_globs(st.files_to_touch)) for st in subtasks]
    for i, (name_a, files_a) in enumerate(items):
        for name_b, files_b in items[i + 1 :]:
            for pa in files_a:
                for pb in files_b:
                    if _patterns_overlap(pa, pb):
                        conflicts.append((name_a, pa, name_b, pb))
    return conflicts


# ──────────────────────────────────────────────────────────────────────────────
#  Topological-Sort (Kahn) mit Batches
# ──────────────────────────────────────────────────────────────────────────────
def topological_batches(subtasks: Sequence) -> List[List]:
    """Sortiert Sub-Tasks topologisch und gibt Batches zurück.

    Eine Batch enthält Sub-Tasks deren Dependencies in vorigen Batches gelöst
    sind und die untereinander disjunkte files_to_touch haben. Bei Overlap
    werden sie auf nachfolgende Batches geschoben.

    Voraussetzung: validate_dag(subtasks) muss leere Liste zurückgegeben haben.
    Bei Zyklen oder unbekannten Refs wird ValueError geworfen.
    """
    if not subtasks:
        return []

    by_name = {st.name: st for st in subtasks}
    remaining = {st.name: set(st.depends_on) for st in subtasks}
    batches: List[List] = []
    completed: set[str] = set()

    while remaining:
        ready = [n for n, deps in remaining.items() if deps <= completed]
        if not ready:
            raise ValueError(
                "DAG-Sortierung fehlgeschlagen — vermutlich Zykel oder "
                "unbekannte depends_on-Ref. Vor topological_batches "
                "validate_dag aufrufen."
            )
        # File-Overlap innerhalb der ready-Menge auflösen: wenn zwei Sub-Tasks
        # überlappen, wird einer (lexikographisch größerer Name) auf die
        # nächste Batch verschoben.
        batch_subtasks = [by_name[n] for n in ready]
        confs = has_file_overlap(batch_subtasks)
        if confs:
            # Sammele alle „loser" — Sub-Tasks die in einem Konflikt-Pair den
            # größeren Namen haben → in nächste Batch.
            losers = {max(a, b) for (a, _, b, _) in confs}
            this_batch = [st for st in batch_subtasks if st.name not in losers]
        else:
            this_batch = batch_subtasks
        if not this_batch:
            # Edge: alle in losers — nimm lexicographisch ersten als Single
            this_batch = [batch_subtasks[0]]
        batches.append(this_batch)
        for st in this_batch:
            completed.add(st.name)
            remaining.pop(st.name)
    return batches
