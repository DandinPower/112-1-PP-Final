import pytest
import math
from src.utils import SparseMatrixTestConfiguration, TestType
from src.extension import ExtensionHandler
import sparse_mm

@pytest.fixture
def handler() -> ExtensionHandler:
    return ExtensionHandler()

@pytest.fixture
def test_configuration():
    config = SparseMatrixTestConfiguration(5, 5, 0.7, 5, 5, 0.7, 4)
    return config

def test_get_extension_config(handler, test_configuration):
    result = handler.get_extension_config(test_configuration)
    assert isinstance(result, sparse_mm.TestConfig)
    assert result.A_col == test_configuration.A_col
    assert result.A_row == test_configuration.A_row
    assert math.isclose(result.A_density, test_configuration.A_density, rel_tol=1e-5)
    assert result.B_col == test_configuration.B_col
    assert result.B_row == test_configuration.B_row
    assert math.isclose(result.B_density, test_configuration.B_density, rel_tol=1e-5)
    assert result.num_threads == test_configuration.num_threads

@pytest.mark.parametrize("test_type", [TestType.BUILTIN, TestType.PARALLEL_STRUCTURE, TestType.OPENMP, TestType.STD_THREAD])
def test_set_extension_config_type(handler, test_configuration, test_type):
    extension_config = handler.get_extension_config(test_configuration)
    handler.set_extension_config_type(extension_config, test_type)
    assert extension_config.test_type == test_type.value