"""开发环境的本地 SQLite artifact repository。

媒体文件已经保存在 `.local_assets/`；此处持久化 artifact 版本、精确 sources 和
asset metadata，使重启后仍能选择旧 preprocess / shot-script 继续分析。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.domain.assets import AssetKind, AssetOrigin, MediaAsset, RightsStatus
from app.domain.errors import CapabilityError, ErrorCode
from app.domain.refs import ArtifactRef, ArtifactType
from app.domain.versioning import ArtifactStatus, ArtifactVersion, StaleMark, validate_transition
from app.storage.local_assets import LocalAssetStore


def _now() -> datetime:
    return datetime.now(UTC)


class SQLiteArtifactRepository:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("""CREATE TABLE IF NOT EXISTS artifacts (
          type TEXT NOT NULL, id TEXT NOT NULL, version INTEGER NOT NULL,
          status TEXT NOT NULL, body TEXT NOT NULL, sources TEXT NOT NULL,
          stale TEXT, created_at TEXT NOT NULL, created_by TEXT NOT NULL,
          PRIMARY KEY(type,id,version))""")
        self._db.commit()

    def create_version(
        self,
        artifact_type: ArtifactType,
        artifact_id: str,
        body: Mapping[str, Any],
        *,
        created_by: str,
        sources: tuple[ArtifactRef, ...] = (),
        status: ArtifactStatus = ArtifactStatus.DRAFT,
    ) -> ArtifactVersion:
        for source in sources:
            self.get(source)
        row = self._db.execute(
            "SELECT COALESCE(MAX(version),0)+1 AS v FROM artifacts WHERE type=? AND id=?",
            (artifact_type.value, artifact_id),
        ).fetchone()
        version = int(row["v"])
        created_at = _now()
        self._db.execute(
            "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?,?,?)",
            (
                artifact_type.value,
                artifact_id,
                version,
                status.value,
                json.dumps(dict(body), ensure_ascii=False, default=str),
                json.dumps(
                    [{"type": s.type.value, "id": s.id, "version": s.version} for s in sources]
                ),
                None,
                created_at.isoformat(),
                created_by,
            ),
        )
        self._db.commit()
        return self.get(ArtifactRef(type=artifact_type, id=artifact_id, version=version))

    def _row_to_version(self, row: sqlite3.Row) -> ArtifactVersion:
        sources = tuple(
            ArtifactRef(type=ArtifactType(s["type"]), id=s["id"], version=s["version"])
            for s in json.loads(row["sources"])
        )
        stale = StaleMark.model_validate(json.loads(row["stale"])) if row["stale"] else None
        return ArtifactVersion(
            type=ArtifactType(row["type"]),
            id=row["id"],
            version=row["version"],
            status=ArtifactStatus(row["status"]),
            body=json.loads(row["body"]),
            sources=sources,
            stale=stale,
            created_at=datetime.fromisoformat(row["created_at"]),
            created_by=row["created_by"],
        )

    def get(self, ref: ArtifactRef) -> ArtifactVersion:
        row = self._db.execute(
            "SELECT * FROM artifacts WHERE type=? AND id=? AND version=?",
            (ref.type.value, ref.id, ref.version),
        ).fetchone()
        if row is None:
            raise CapabilityError(ErrorCode.ARTIFACT_NOT_FOUND, f"产物版本不存在：{ref}")
        return self._row_to_version(row)

    def latest(self, artifact_type: ArtifactType, artifact_id: str) -> ArtifactVersion:
        row = self._db.execute(
            "SELECT * FROM artifacts WHERE type=? AND id=? ORDER BY version DESC LIMIT 1",
            (artifact_type.value, artifact_id),
        ).fetchone()
        if row is None:
            raise CapabilityError(
                ErrorCode.ARTIFACT_NOT_FOUND, f"产物不存在：{artifact_type}:{artifact_id}"
            )
        return self._row_to_version(row)

    def history(self, artifact_type: ArtifactType, artifact_id: str) -> tuple[ArtifactVersion, ...]:
        rows = self._db.execute(
            "SELECT * FROM artifacts WHERE type=? AND id=? ORDER BY version",
            (artifact_type.value, artifact_id),
        ).fetchall()
        return tuple(self._row_to_version(row) for row in rows)

    def dependents(self, ref: ArtifactRef) -> tuple[ArtifactRef, ...]:
        rows = self._db.execute("SELECT type,id,version,sources FROM artifacts").fetchall()
        result: list[ArtifactRef] = []
        for row in rows:
            if any(
                s == {"type": ref.type.value, "id": ref.id, "version": ref.version}
                for s in json.loads(row["sources"])
            ):
                result.append(
                    ArtifactRef(
                        type=ArtifactType(row["type"]), id=row["id"], version=row["version"]
                    )
                )
        return tuple(result)

    def list_by_type(self, artifact_type: ArtifactType) -> tuple[ArtifactVersion, ...]:
        rows = self._db.execute(
            "SELECT * FROM artifacts WHERE type=? ORDER BY created_at DESC", (artifact_type.value,)
        ).fetchall()
        return tuple(self._row_to_version(row) for row in rows)

    def transition(self, ref: ArtifactRef, target: ArtifactStatus) -> ArtifactVersion:
        current = self.get(ref)
        validate_transition(current.status, target)
        self._db.execute(
            "UPDATE artifacts SET status=? WHERE type=? AND id=? AND version=?",
            (target.value, ref.type.value, ref.id, ref.version),
        )
        self._db.commit()
        return self.get(ref)

    def mark_stale(self, ref: ArtifactRef, mark: StaleMark) -> ArtifactVersion:
        current = self.get(ref)
        if current.stale is None:
            self._db.execute(
                "UPDATE artifacts SET stale=? WHERE type=? AND id=? AND version=?",
                (mark.model_dump_json(), ref.type.value, ref.id, ref.version),
            )
            self._db.commit()
        return self.get(ref)


class PersistentLocalAssetStore(LocalAssetStore):
    """LocalAssetStore + JSON metadata index, so asset_id survives restart."""

    def __init__(self, root: Path, metadata_path: Path) -> None:
        super().__init__(root)
        self._metadata_path = metadata_path
        if metadata_path.exists():
            data = json.loads(metadata_path.read_text())
            self._assets = {
                asset_id: MediaAsset.model_validate(value) for asset_id, value in data.items()
            }

    def _save(self) -> None:
        self._metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata_path.write_text(
            json.dumps(
                {key: value.model_dump(mode="json") for key, value in self._assets.items()},
                ensure_ascii=False,
            )
        )

    def put(
        self,
        path: Path,
        *,
        kind: AssetKind,
        origin: AssetOrigin,
        mime_type: str,
        created_by: str,
        rights_status: RightsStatus = RightsStatus.UNKNOWN,
        rights_note: str | None = None,
        source_url: str | None = None,
        derived_from: str | None = None,
    ) -> MediaAsset:
        asset = super().put(
            path,
            kind=kind,
            origin=origin,
            mime_type=mime_type,
            created_by=created_by,
            rights_status=rights_status,
            rights_note=rights_note,
            source_url=source_url,
            derived_from=derived_from,
        )
        self._save()
        return asset

    def soft_delete(self, asset_id: str, *, deleted_by: str) -> MediaAsset:
        asset = super().soft_delete(asset_id, deleted_by=deleted_by)
        self._save()
        return asset
