"""Tests for dataset class."""
import pytest


class TestAntisemitismDataset:
    """Placeholder tests for dataset — requires tokenizer to run fully."""

    def test_import(self):
        """Test that dataset module can be imported."""
        pytest.importorskip("transformers")
        from src.data.dataset import AntisemitismDataset
        assert AntisemitismDataset is not None

    def test_keyword_masking_import(self):
        """Test that keyword masking module can be imported."""
        from src.data.keyword_masking import keyword_sensitivity_test
        assert keyword_sensitivity_test is not None
