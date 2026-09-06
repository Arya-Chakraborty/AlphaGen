from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from molecule_gen.paths import ensure_processed_dir, mordred_cache_path


class DescriptorCache:
    """SQLite cache of descriptor vectors keyed by InChIKey."""

    def __init__(self, path: Path | None = None) -> None:
        ensure_processed_dir()
        self.path = path or mordred_cache_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS descriptors (
                inchikey TEXT PRIMARY KEY,
                smiles TEXT NOT NULL,
                n_descriptors INTEGER NOT NULL,
                payload TEXT NOT NULL,
                computed_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS skipped (
                inchikey TEXT PRIMARY KEY,
                smiles TEXT NOT NULL,
                reason TEXT NOT NULL,
                skipped_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> DescriptorCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM descriptors").fetchone()
        return int(row[0])

    def has(self, inchikey: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM descriptors WHERE inchikey = ? LIMIT 1",
            (inchikey,),
        ).fetchone()
        return row is not None

    def inchikeys(self) -> set[str]:
        rows = self._conn.execute("SELECT inchikey FROM descriptors").fetchall()
        return {key for (key,) in rows}

    def skipped_inchikeys(self) -> set[str]:
        rows = self._conn.execute("SELECT inchikey FROM skipped").fetchall()
        return {key for (key,) in rows}

    def mark_skipped(self, inchikey: str, smiles: str, reason: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO skipped (inchikey, smiles, reason, skipped_at)
            VALUES (?, ?, ?, ?)
            """,
            (inchikey, smiles, reason, now),
        )
        self._conn.commit()

    def get(self, inchikey: str) -> dict[str, float | None] | None:
        row = self._conn.execute(
            "SELECT payload FROM descriptors WHERE inchikey = ?",
            (inchikey,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def get_many(self, inchikeys: list[str]) -> dict[str, dict[str, float | None]]:
        found: dict[str, dict[str, float | None]] = {}
        unique = list(dict.fromkeys(inchikeys))
        chunk = 400
        for i in range(0, len(unique), chunk):
            part = unique[i : i + chunk]
            placeholders = ",".join("?" * len(part))
            rows = self._conn.execute(
                f"SELECT inchikey, payload FROM descriptors WHERE inchikey IN ({placeholders})",
                part,
            ).fetchall()
            for key, payload in rows:
                found[key] = json.loads(payload)
        return found

    def put(
        self,
        inchikey: str,
        smiles: str,
        descriptors: dict[str, float | None],
    ) -> None:
        self.put_many([(inchikey, smiles, descriptors)])

    def put_many(
        self,
        rows: list[tuple[str, str, dict[str, float | None]]],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO descriptors
                (inchikey, smiles, n_descriptors, payload, computed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    inchikey,
                    smiles,
                    len(descriptors),
                    json.dumps(descriptors, sort_keys=True),
                    now,
                )
                for inchikey, smiles, descriptors in rows
            ],
        )
        self._conn.commit()
