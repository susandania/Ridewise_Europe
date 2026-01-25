from __future__ import annotations

from pathlib import Path
import pickle
import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def artifacts_dir(repo_root: Path) -> Path:
    return repo_root / "models/artifacts"


@pytest.fixture(scope="session")
def churn_artifact_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / "churn_model.pkl"


@pytest.fixture(scope="session")
def segment_artifact_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / "segmentation_model.pkl"


@pytest.fixture(scope="session")
def churn_artifact(churn_artifact_path: Path) -> dict:
    with open(churn_artifact_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def segment_artifact(segment_artifact_path: Path) -> dict:
    with open(segment_artifact_path, "rb") as f:
        return pickle.load(f)
