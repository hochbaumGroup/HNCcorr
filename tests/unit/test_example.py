from conftest import TEST_DATA_DIR

from hnccorr.example import load_example_data


def test_example_data_loader():
    data = load_example_data(filedir=TEST_DATA_DIR, memmap=True)
    assert data.shape == (800, 512, 512)
