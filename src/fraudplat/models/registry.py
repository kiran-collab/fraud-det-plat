"""Filesystem-backed model registry.

A "version" is an immutable directory of artifacts plus a manifest recording
what produced it - data window, row counts, metrics, feature list, git SHA. The
``current`` symlink is the only mutable thing, so promotion and rollback are
one atomic operation and the audit trail is the directory listing itself.

Backed by S3 in EKS (``FP_MODEL_DIR=s3://...`` via s3fs) and by the local
filesystem in dev and CI. The manifest is what the Model Risk Management pack
in ``docs/model_governance.md`` is generated from.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fraudplat.config import SETTINGS
from fraudplat.features.transforms import FEATURE_NAMES, MerchantProfile
from fraudplat.models.ensemble import EnsembleScorer
from fraudplat.models.iforest import IsolationForestScorer
from fraudplat.models.lgbm import SupervisedModel
from fraudplat.models.transformer import SequenceAnomalyModel


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return os.environ.get("GIT_SHA", "unknown")


@dataclass
class ModelBundle:
    """Everything needed to score one transaction, loaded together."""

    supervised: SupervisedModel
    iforest: IsolationForestScorer
    sequence: SequenceAnomalyModel
    ensemble: EnsembleScorer
    merchant_profile: MerchantProfile
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> str:
        return str(self.manifest.get("version", "unversioned"))


class ModelRegistry:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or SETTINGS.paths.models)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths -----------------------------------------------------------
    def version_dir(self, version: str) -> Path:
        return self.root / version

    def current_dir(self) -> Path:
        link = self.root / "current"
        if link.exists():
            return link.resolve()
        versions = self.list_versions()
        if not versions:
            raise FileNotFoundError(
                f"no model versions under {self.root}. Run scripts/train.py first."
            )
        return self.version_dir(versions[-1])

    def list_versions(self) -> list[str]:
        return sorted(
            p.name for p in self.root.iterdir()
            if p.is_dir() and p.name != "current" and (p / "manifest.json").exists()
        )

    @staticmethod
    def new_version() -> str:
        return datetime.now(UTC).strftime("v%Y%m%d-%H%M%S")

    # -- write -----------------------------------------------------------
    def save(self, bundle: ModelBundle, version: str | None = None, promote: bool = True) -> Path:
        version = version or self.new_version()
        d = self.version_dir(version)
        d.mkdir(parents=True, exist_ok=True)

        bundle.supervised.save(d / "supervised.lgb")
        bundle.iforest.save(d / "iforest.joblib")
        bundle.sequence.save(d / "sequence.pt")
        bundle.ensemble.save(d / "calibrator.joblib")
        (d / "merchant_profile.json").write_text(json.dumps(bundle.merchant_profile.to_dict()))

        # Registry-owned fields are written *after* the caller's manifest, not
        # before. Spreading the caller's dict last let a stray "version" key in
        # training metadata silently overwrite the directory's real version, so
        # the artifact on disk and the version recorded inside it disagreed -
        # which defeats the entire point of an immutable versioned registry.
        manifest = {
            **bundle.manifest,
            "version": version,
            "created_at": datetime.now(UTC).isoformat(),
            "git_sha": _git_sha(),
            "feature_names": FEATURE_NAMES,
            "feature_count": len(FEATURE_NAMES),
        }
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

        if promote:
            self.promote(version)
        return d

    def promote(self, version: str) -> None:
        """Atomically repoint ``current``. Rollback is the same call with an
        older version, which is why nothing else in the codebase reads a
        version string directly."""
        target = self.version_dir(version)
        if not (target / "manifest.json").exists():
            raise FileNotFoundError(f"unknown model version: {version}")
        link = self.root / "current"
        tmp = self.root / ".current.tmp"
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        tmp.symlink_to(target, target_is_directory=True)
        os.replace(tmp, link)

    # -- read ------------------------------------------------------------
    def load(self, version: str | None = None) -> ModelBundle:
        d = self.version_dir(version) if version else self.current_dir()
        manifest = json.loads((d / "manifest.json").read_text())

        expected = manifest.get("feature_names", [])
        if expected and expected != FEATURE_NAMES:
            raise RuntimeError(
                "feature contract mismatch: the code's FEATURE_NAMES no longer matches "
                f"model {manifest.get('version')}. Retrain rather than scoring with a "
                "misaligned vector."
            )

        return ModelBundle(
            supervised=SupervisedModel.load(d / "supervised.lgb"),
            iforest=IsolationForestScorer.load(d / "iforest.joblib"),
            sequence=SequenceAnomalyModel.load(d / "sequence.pt"),
            ensemble=EnsembleScorer.load(d / "calibrator.joblib"),
            merchant_profile=MerchantProfile.from_dict(
                json.loads((d / "merchant_profile.json").read_text())
            ),
            manifest=manifest,
        )
