"""Real on-disk CAS/history test adapter, NOT production checkpoint storage."""

import hashlib
import sqlite3

from bjj_analysis_core.execution import StepCheckpointV1


class DiskJournal:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS heads (id TEXT PRIMARY KEY, revision INTEGER, body TEXT, sha TEXT)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS history (id TEXT, revision INTEGER, body TEXT, sha TEXT, PRIMARY KEY(id, revision))"
        )
        self.db.commit()

    async def read(self, step_id):
        row = self.db.execute(
            "SELECT body, sha FROM heads WHERE id=?", (str(step_id),)
        ).fetchone()
        if row is None:
            return None
        if hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
            raise ValueError("checkpoint checksum mismatch")
        return StepCheckpointV1.model_validate_json(row[0])

    async def compare_and_swap(self, expected_revision, checkpoint):
        body = checkpoint.model_dump_json()
        values = (
            str(checkpoint.step_id),
            checkpoint.revision,
            body,
            hashlib.sha256(body.encode()).hexdigest(),
        )
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                "SELECT revision FROM heads WHERE id=?", (values[0],)
            ).fetchone()
            if (None if row is None else row[0]) != expected_revision:
                self.db.rollback()
                return False
            assert checkpoint.revision == (expected_revision or 0) + 1
            self.db.execute("INSERT INTO history VALUES (?,?,?,?)", values)
            self.db.execute("INSERT OR REPLACE INTO heads VALUES (?,?,?,?)", values)
            self.db.commit()
            return True
        except BaseException:
            self.db.rollback()
            raise

    def records(self, history=False):
        table = "history" if history else "heads"
        return [
            StepCheckpointV1.model_validate_json(r[0])
            for r in self.db.execute(f"SELECT body FROM {table} ORDER BY rowid")
        ]

    def close(self):
        self.db.close()
