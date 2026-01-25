def test_churn_artifact_file_exists(churn_artifact_path):
    assert churn_artifact_path.exists(), f"Missing: {churn_artifact_path}"


def test_segmentation_artifact_file_exists(segment_artifact_path):
    assert segment_artifact_path.exists(), f"Missing: {segment_artifact_path}"


    # Expected keys
def test_churn_artifact_contract(churn_artifact):
    assert hasattr(churn_artifact, "__len__")
    assert len(churn_artifact) > 0

    assert all(isinstance(x, (str,)) and x.strip() for x in churn_artifact)


    # Segmentation expected keys
def test_segmentation_artifact_contract(segment_artifact):
    assert hasattr(segment_artifact, "__len__")
    assert len(segment_artifact) > 0

    assert all(isinstance(x, (str,)) and x.strip() for x in segment_artifact)

