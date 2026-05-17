import pytest
import pandas as pd
import os
import shutil

@pytest.fixture
def valid_csv_data(tmp_path):
    """Create a valid CSV for testing."""
    df = pd.DataFrame({
        "my_prompt": ["This is a long enough prompt for testing 1", "This is a long enough prompt for testing 2", "This is a long enough prompt for testing 3"],
        "my_chosen": ["This is a good and helpful response that meets the length requirements.", "Another good response that is also quite long enough.", "Yes, this is a third good response for our test suite."],
        "my_rejected": ["This is a bad response that we want the model to avoid in training.", "A very poor quality response that should be rejected by DPO.", "Not a good answer at all, please do not use this one."]
    })
    path = tmp_path / "valid_data.csv"
    df.to_csv(path, index=False)
    return str(path)

@pytest.fixture
def invalid_csv_data(tmp_path):
    """Create an invalid CSV (missing column)."""
    df = pd.DataFrame({
        "my_prompt": ["Q1"],
        "wrong_column": ["A1"]
    })
    path = tmp_path / "invalid_data.csv"
    df.to_csv(path, index=False)
    return str(path)

@pytest.fixture(autouse=True)
def cleanup_outputs():
    """Cleanup outputs directory after tests."""
    yield
    if os.path.exists("./outputs_test"):
        shutil.rmtree("./outputs_test")
    if os.path.exists("./logs_test"):
        shutil.rmtree("./logs_test")
