"""
Tests for the GEO data loader module.

Tests cover:
- GEODataLoader initialization and validation
- URL building logic
- File extraction handling
- Methylation matrix loading
- Metadata parsing
"""

import pytest
import tempfile
import gzip
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from src.data.loader import GEODataLoader, load_series_matrix


class TestGEODataLoaderInit:
    """Tests for GEODataLoader initialization."""

    def test_valid_accession_format(self):
        """Test that valid GEO accession IDs are accepted."""
        loader = GEODataLoader('GSE171140')
        assert loader.geo_accession == 'GSE171140'

    def test_invalid_accession_format(self):
        """Test that invalid accession formats raise ValueError."""
        with pytest.raises(ValueError, match="Invalid GEO accession format"):
            GEODataLoader('INVALID123')

    def test_invalid_accession_wrong_prefix(self):
        """Test that accession without GSE prefix raises ValueError."""
        with pytest.raises(ValueError, match="Invalid GEO accession format"):
            GEODataLoader('GPL171140')

    def test_default_data_dir(self):
        """Test that default data directory is created correctly."""
        loader = GEODataLoader('GSE171140')
        assert 'data' in str(loader.data_dir)
        assert 'raw' in str(loader.data_dir)

    def test_custom_data_dir(self, tmp_path):
        """Test that custom data directory is used."""
        loader = GEODataLoader('GSE171140', data_dir=tmp_path)
        assert loader.data_dir == tmp_path

    def test_data_dir_created(self, tmp_path):
        """Test that data directory is created if it doesn't exist."""
        custom_dir = tmp_path / 'custom' / 'data'
        loader = GEODataLoader('GSE171140', data_dir=custom_dir)
        assert custom_dir.exists()

    def test_default_metadata_lines(self):
        """Test default metadata lines count."""
        loader = GEODataLoader('GSE171140')
        assert loader.metadata_lines == 74

    def test_custom_metadata_lines(self):
        """Test custom metadata lines count."""
        loader = GEODataLoader('GSE171140', metadata_lines=100)
        assert loader.metadata_lines == 100


class TestGEODataLoaderURLBuilding:
    """Tests for URL building functionality."""

    def test_url_format(self):
        """Test that download URL is built correctly."""
        loader = GEODataLoader('GSE171140')
        url = loader._build_download_url()

        assert 'ftp.ncbi.nlm.nih.gov' in url
        assert 'GSE171' in url
        assert 'GSE171140' in url
        assert 'series_matrix.txt.gz' in url

    def test_url_different_accessions(self):
        """Test URL building for different accession ranges."""
        loader1 = GEODataLoader('GSE12345')
        url1 = loader1._build_download_url()
        assert 'GSE12' in url1

        loader2 = GEODataLoader('GSE123456')
        url2 = loader2._build_download_url()
        assert 'GSE123' in url2


class TestGEODataLoaderFileOperations:
    """Tests for file operations (extraction, loading)."""

    def test_extract_gz_file_not_found(self, tmp_path):
        """Test that extracting non-existent file raises FileNotFoundError."""
        loader = GEODataLoader('GSE171140', data_dir=tmp_path)

        with pytest.raises(FileNotFoundError):
            loader.extract_gz_file()

    def test_extract_gz_file_success(self, tmp_path):
        """Test successful extraction of gzip file."""
        loader = GEODataLoader('GSE171140', data_dir=tmp_path)

        # Create a mock gzipped file
        gz_path = loader._gz_path
        content = b"test content"
        with gzip.open(gz_path, 'wb') as f:
            f.write(content)

        # Extract and verify
        extracted_path = loader.extract_gz_file()
        assert extracted_path.exists()
        with open(extracted_path, 'rb') as f:
            assert f.read() == content

    def test_extract_already_exists_no_force(self, tmp_path):
        """Test that extraction is skipped when file exists and force=False."""
        loader = GEODataLoader('GSE171140', data_dir=tmp_path)

        # Create both gz and extracted file
        gz_path = loader._gz_path
        with gzip.open(gz_path, 'wb') as f:
            f.write(b"new content")

        extracted_path = loader._extracted_path
        with open(extracted_path, 'wb') as f:
            f.write(b"old content")

        # Should return existing file without re-extraction
        result_path = loader.extract_gz_file(force=False)
        assert result_path == extracted_path
        with open(result_path, 'rb') as f:
            assert f.read() == b"old content"

    def test_load_methylation_matrix_file_not_found(self, tmp_path):
        """Test loading methylation matrix when file doesn't exist."""
        loader = GEODataLoader('GSE171140', data_dir=tmp_path)

        with pytest.raises(FileNotFoundError):
            loader.load_methylation_matrix()

    def test_load_methylation_matrix_basic(self, tmp_path):
        """Test loading a basic methylation matrix."""
        loader = GEODataLoader('GSE171140', data_dir=tmp_path, metadata_lines=2)

        # Create a mock series matrix file
        matrix_content = """!Series_title	"Test Series"
!Series_type	"Methylation"
"ID_REF"	"GSM0000001"	"GSM0000002"	"GSM0000003"
"cg00000001"	0.5	0.6	0.7
"cg00000002"	0.3	0.4	0.5
"cg00000003"	0.8	0.7	0.6
!series_matrix_table_end"""

        matrix_path = loader._extracted_path
        with open(matrix_path, 'w') as f:
            f.write(matrix_content)

        df = loader.load_methylation_matrix()

        # Verify structure
        assert df.shape == (3, 3)
        assert 'cg00000001' in df.index
        assert 'GSM0000001' in df.columns
        assert df.index.name == 'probe_id'

    def test_preview_file(self, tmp_path):
        """Test file preview functionality."""
        loader = GEODataLoader('GSE171140', data_dir=tmp_path)

        # Create a test file
        content = "\n".join([f"Line {i}" for i in range(100)])
        with open(loader._extracted_path, 'w') as f:
            f.write(content)

        lines = loader.preview_file(10)
        assert len(lines) == 10
        assert lines[0] == "Line 0"
        assert lines[9] == "Line 9"


class TestGEODataLoaderMetadata:
    """Tests for metadata extraction."""

    def test_get_metadata_basic(self, tmp_path):
        """Test basic metadata extraction."""
        loader = GEODataLoader('GSE171140', data_dir=tmp_path, metadata_lines=5)

        metadata_content = """!Sample_title	"Sample 1"	"Sample 2"
!Sample_geo_accession	"GSM0000001"	"GSM0000002"
!Sample_source_name_ch1	"Blood"	"Blood"
!Sample_characteristics_ch1	"age: 25"	"age: 30"
!Series_type	"Expression"
"ID_REF"	"GSM0000001"	"GSM0000002"
"cg00000001"	0.5	0.6"""

        with open(loader._extracted_path, 'w') as f:
            f.write(metadata_content)

        metadata = loader.get_metadata()

        assert 'Sample_title' in metadata
        assert 'Sample_geo_accession' in metadata
        assert len(metadata['Sample_title']) == 2
        assert metadata['Sample_geo_accession'][0] == 'GSM0000001'

    def test_get_metadata_caching(self, tmp_path):
        """Test that metadata is cached after first load."""
        loader = GEODataLoader('GSE171140', data_dir=tmp_path, metadata_lines=2)

        content = """!Sample_title	"Sample 1"
"ID_REF"	"GSM0000001"
"cg00000001"	0.5"""

        with open(loader._extracted_path, 'w') as f:
            f.write(content)

        # First call
        metadata1 = loader.get_metadata()

        # Modify file
        with open(loader._extracted_path, 'w') as f:
            f.write("""!Sample_title	"Sample 2"
"ID_REF"	"GSM0000001"
"cg00000001"	0.5""")

        # Second call should return cached version
        metadata2 = loader.get_metadata()
        assert metadata2['Sample_title'][0] == 'Sample 1'

        # Force reload should get new data
        metadata3 = loader.get_metadata(force_reload=True)
        assert metadata3['Sample_title'][0] == 'Sample 2'


class TestLoadSeriesMatrix:
    """Tests for the load_series_matrix convenience function."""

    def test_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_series_matrix(tmp_path / "nonexistent.txt")

    def test_basic_loading(self, tmp_path):
        """Test basic series matrix loading."""
        matrix_path = tmp_path / "test_matrix.txt"

        content = """!Sample_title	"S1"	"S2"
!Sample_geo_accession	"GSM1"	"GSM2"
"ID_REF"	"GSM1"	"GSM2"
"cg001"	0.5	0.6
"cg002"	0.3	0.4"""

        with open(matrix_path, 'w') as f:
            f.write(content)

        data, metadata = load_series_matrix(matrix_path, skip_rows=2)

        assert data.shape == (2, 2)
        assert 'Sample_title' in metadata
        assert len(metadata['Sample_geo_accession']) == 2
