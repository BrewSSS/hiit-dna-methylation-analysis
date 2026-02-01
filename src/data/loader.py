"""
GEO Data Loader module for downloading and loading methylation data.

This module provides functionality to download GEO series matrix files,
extract compressed data, and load methylation beta values into DataFrames.
"""

import os
import gzip
import shutil
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any, Union

import requests
import pandas as pd
import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

logger = logging.getLogger(__name__)


class GEODataLoader:
    """
    A class for downloading and loading GEO series matrix data.

    This class handles the complete workflow of acquiring GEO methylation data:
    downloading compressed files, extracting them, and parsing the series matrix
    format into usable DataFrames.

    Attributes:
        geo_accession: The GEO accession ID (e.g., 'GSE171140').
        data_dir: Directory path for storing downloaded data.
        metadata_lines: Number of metadata lines to skip in series matrix files.

    Example:
        >>> loader = GEODataLoader('GSE171140', data_dir='./data/raw')
        >>> loader.download_series_matrix()
        >>> methylation_data = loader.load_methylation_matrix()
        >>> metadata = loader.get_metadata()
    """

    # GEO FTP URL template
    GEO_FTP_TEMPLATE = (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}nnn/{accession}/matrix/"
        "{accession}_series_matrix.txt.gz"
    )

    def __init__(
        self,
        geo_accession: str,
        data_dir: Optional[Union[str, Path]] = None,
        metadata_lines: int = 74
    ) -> None:
        """
        Initialize the GEO data loader.

        Args:
            geo_accession: GEO accession ID (e.g., 'GSE171140').
            data_dir: Directory for storing data files. If None, uses current
                working directory with 'data/raw' subdirectory.
            metadata_lines: Number of metadata lines at the beginning of
                series matrix files. Default is 74 for standard GEO format.

        Raises:
            ValueError: If geo_accession format is invalid.
        """
        if not geo_accession.startswith('GSE'):
            raise ValueError(f"Invalid GEO accession format: {geo_accession}")

        self.geo_accession = geo_accession
        self.metadata_lines = metadata_lines

        if data_dir is None:
            self.data_dir = Path.cwd() / 'data' / 'raw'
        else:
            self.data_dir = Path(data_dir)

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self._gz_path = self.data_dir / f"{geo_accession}_series_matrix.txt.gz"
        self._extracted_path = self.data_dir / f"{geo_accession}_series_matrix.txt"

        # Cache for metadata
        self._metadata_cache: Optional[Dict[str, List[str]]] = None

    @property
    def series_matrix_path(self) -> Path:
        """Path to the extracted series matrix file."""
        return self._extracted_path

    @property
    def gz_path(self) -> Path:
        """Path to the compressed series matrix file."""
        return self._gz_path

    def _build_download_url(self) -> str:
        """
        Build the download URL for the GEO series matrix file.

        Returns:
            The complete URL for downloading the series matrix file.
        """
        prefix = self.geo_accession[:len(self.geo_accession) - 3]
        return self.GEO_FTP_TEMPLATE.format(
            prefix=prefix,
            accession=self.geo_accession
        )

    def download_series_matrix(
        self,
        url: Optional[str] = None,
        force: bool = False,
        chunk_size: int = 8192
    ) -> Path:
        """
        Download the series matrix file from GEO with progress tracking.

        Args:
            url: Custom URL for downloading. If None, uses the standard GEO FTP URL.
            force: If True, re-download even if file exists.
            chunk_size: Size of download chunks in bytes.

        Returns:
            Path to the downloaded compressed file.

        Raises:
            requests.RequestException: If download fails.
            IOError: If file writing fails.

        Example:
            >>> loader = GEODataLoader('GSE171140')
            >>> gz_path = loader.download_series_matrix()
            >>> print(f"Downloaded to: {gz_path}")
        """
        if self._gz_path.exists() and not force:
            logger.info(f"File already exists: {self._gz_path}")
            return self._gz_path

        download_url = url or self._build_download_url()
        logger.info(f"Starting download: {download_url}")

        start_time = time.time()

        with requests.get(download_url, stream=True) as response:
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            if TQDM_AVAILABLE and total_size > 0:
                progress_bar = tqdm(
                    desc=self._gz_path.name,
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                )
            else:
                progress_bar = None

            with open(self._gz_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        if progress_bar:
                            progress_bar.update(len(chunk))

            if progress_bar:
                progress_bar.close()

        elapsed = time.time() - start_time
        file_size_mb = self._gz_path.stat().st_size / (1024 * 1024)
        logger.info(f"Download complete: {file_size_mb:.2f} MB in {elapsed:.2f} seconds")

        return self._gz_path

    def extract_gz_file(
        self,
        gz_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        force: bool = False
    ) -> Path:
        """
        Extract a gzip-compressed file.

        Args:
            gz_path: Path to the compressed file. If None, uses the default path.
            output_path: Path for the extracted file. If None, removes .gz extension.
            force: If True, re-extract even if output file exists.

        Returns:
            Path to the extracted file.

        Raises:
            FileNotFoundError: If the compressed file does not exist.
            IOError: If extraction fails.

        Example:
            >>> loader = GEODataLoader('GSE171140')
            >>> extracted = loader.extract_gz_file()
            >>> print(f"Extracted to: {extracted}")
        """
        gz_path = Path(gz_path) if gz_path else self._gz_path

        if output_path is None:
            output_path = Path(str(gz_path).replace('.gz', ''))
        else:
            output_path = Path(output_path)

        if output_path.exists() and not force:
            logger.info(f"Extracted file already exists: {output_path}")
            return output_path

        if not gz_path.exists():
            raise FileNotFoundError(f"Compressed file not found: {gz_path}")

        logger.info(f"Extracting: {gz_path}")
        start_time = time.time()

        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        elapsed = time.time() - start_time
        logger.info(f"Extraction complete in {elapsed:.2f} seconds")

        # Update internal path
        self._extracted_path = output_path

        return output_path

    def load_methylation_matrix(
        self,
        skip_rows: Optional[int] = None,
        usecols: Optional[List[str]] = None,
        nrows: Optional[int] = None,
        dtype: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Load the methylation beta values matrix from the series matrix file.

        The series matrix file contains metadata lines followed by a tab-separated
        data matrix where rows are CpG probes and columns are samples.

        Args:
            skip_rows: Number of rows to skip (metadata). If None, uses default.
            usecols: List of column names to load. If None, loads all columns.
            nrows: Number of data rows to load. If None, loads all rows.
            dtype: Dictionary of column data types.

        Returns:
            DataFrame with CpG probe IDs as index and sample IDs as columns.
            Values are beta values (methylation levels between 0 and 1).

        Raises:
            FileNotFoundError: If series matrix file does not exist.
            ValueError: If file format is unexpected.

        Example:
            >>> loader = GEODataLoader('GSE171140')
            >>> data = loader.load_methylation_matrix()
            >>> print(f"Loaded {data.shape[0]} CpGs x {data.shape[1]} samples")
        """
        if not self._extracted_path.exists():
            raise FileNotFoundError(
                f"Series matrix file not found: {self._extracted_path}. "
                "Please download and extract the file first."
            )

        skip_rows = skip_rows if skip_rows is not None else self.metadata_lines

        logger.info(f"Loading methylation matrix from {self._extracted_path}")
        logger.info(f"Skipping {skip_rows} metadata rows")

        start_time = time.time()

        # Read the data matrix
        df = pd.read_csv(
            self._extracted_path,
            sep='\t',
            skiprows=skip_rows,
            index_col=0,
            usecols=usecols,
            nrows=nrows,
            dtype=dtype,
            low_memory=False
        )

        # Clean up column names (remove quotes if present)
        df.columns = [col.strip('"') for col in df.columns]
        df.index.name = 'probe_id'

        # Remove the last row if it contains "!series_matrix_table_end"
        if df.index[-1] == '!series_matrix_table_end':
            df = df.iloc[:-1]

        elapsed = time.time() - start_time
        logger.info(f"Loaded {df.shape[0]} probes x {df.shape[1]} samples in {elapsed:.2f}s")

        return df

    def get_metadata(
        self,
        force_reload: bool = False
    ) -> Dict[str, List[str]]:
        """
        Extract sample metadata from the series matrix file.

        Parses the metadata lines at the beginning of the series matrix file
        and returns structured metadata for each sample.

        Args:
            force_reload: If True, re-read metadata even if cached.

        Returns:
            Dictionary with metadata field names as keys and lists of values
            for each sample as values.

        Raises:
            FileNotFoundError: If series matrix file does not exist.

        Example:
            >>> loader = GEODataLoader('GSE171140')
            >>> metadata = loader.get_metadata()
            >>> print(f"Sample titles: {metadata['Sample_title'][:5]}")
        """
        if self._metadata_cache is not None and not force_reload:
            return self._metadata_cache

        if not self._extracted_path.exists():
            raise FileNotFoundError(
                f"Series matrix file not found: {self._extracted_path}"
            )

        metadata: Dict[str, List[str]] = {}

        try:
            with open(self._extracted_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= self.metadata_lines:
                        break

                    if line.startswith('!Sample_'):
                        parts = line.strip().split('\t')
                        field_name = parts[0].replace('!', '')
                        values = [v.strip('"') for v in parts[1:]]

                        if field_name in metadata:
                            # Append to existing field (e.g., characteristics)
                            for j, val in enumerate(values):
                                if j < len(metadata[field_name]):
                                    metadata[field_name][j] += f"; {val}"
                        else:
                            metadata[field_name] = values

        except UnicodeDecodeError:
            # Fallback to latin-1 encoding
            with open(self._extracted_path, 'r', encoding='latin-1') as f:
                for i, line in enumerate(f):
                    if i >= self.metadata_lines:
                        break

                    if line.startswith('!Sample_'):
                        parts = line.strip().split('\t')
                        field_name = parts[0].replace('!', '')
                        values = [v.strip('"') for v in parts[1:]]

                        if field_name not in metadata:
                            metadata[field_name] = values

        self._metadata_cache = metadata
        logger.info(f"Extracted metadata for {len(metadata.get('Sample_geo_accession', []))} samples")

        return metadata

    def preview_file(self, num_lines: int = 50) -> List[str]:
        """
        Preview the first lines of the series matrix file.

        Args:
            num_lines: Number of lines to preview.

        Returns:
            List of lines from the file.

        Example:
            >>> loader = GEODataLoader('GSE171140')
            >>> lines = loader.preview_file(10)
            >>> for line in lines:
            ...     print(line)
        """
        if not self._extracted_path.exists():
            raise FileNotFoundError(f"File not found: {self._extracted_path}")

        lines = []
        with open(self._extracted_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                lines.append(line.strip())

        return lines


def load_series_matrix(
    file_path: Union[str, Path],
    skip_rows: int = 74
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Load a GEO series matrix file and return data and metadata.

    This is a convenience function for loading a series matrix file
    without creating a GEODataLoader instance.

    Args:
        file_path: Path to the series matrix text file.
        skip_rows: Number of metadata rows to skip when loading data.

    Returns:
        Tuple of (data_df, metadata_dict) where:
        - data_df: DataFrame with CpG probes as rows and samples as columns
        - metadata_dict: Dictionary of sample metadata

    Raises:
        FileNotFoundError: If file does not exist.

    Example:
        >>> data, metadata = load_series_matrix('GSE171140_series_matrix.txt')
        >>> print(f"Shape: {data.shape}")
        >>> print(f"Samples: {metadata['Sample_geo_accession'][:5]}")
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Extract metadata
    metadata: Dict[str, List[str]] = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= skip_rows:
                break

            if line.startswith('!Sample_'):
                parts = line.strip().split('\t')
                field_name = parts[0].replace('!', '')
                values = [v.strip('"') for v in parts[1:]]

                if field_name not in metadata:
                    metadata[field_name] = values

    # Load data matrix
    df = pd.read_csv(
        file_path,
        sep='\t',
        skiprows=skip_rows,
        index_col=0,
        low_memory=False
    )

    df.columns = [col.strip('"') for col in df.columns]
    df.index.name = 'probe_id'

    # Remove end marker if present
    if df.index[-1] == '!series_matrix_table_end':
        df = df.iloc[:-1]

    return df, metadata
