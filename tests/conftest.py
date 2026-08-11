"""pytest configuration for iNDI_microscopy_production tests."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: marks tests that require real data")
    config.addinivalue_line("markers", "slow: marks tests that are slow to run")
