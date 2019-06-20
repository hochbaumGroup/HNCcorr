import pytest
import os

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data")


@pytest.fixture
def dummy():
    return "Dummy"
