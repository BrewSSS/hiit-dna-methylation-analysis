"""
Sample mapping module for HIIT methylation study.

This module provides functionality for parsing sample metadata,
creating sample mappings, and generating classification labels
for binary, multiclass, and time-series analyses.
"""

import re
import logging
from typing import Optional, Dict, List, Tuple, Union, Any
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Regex patterns for parsing sample information
TIMEPOINT_PATTERNS = {
    'baseline': {
        'title_patterns': ['PRE'],
        'source_patterns': ['baseline'],
        'characteristic_patterns': ['baseline']
    },
    '4w_hiit': {
        'title_patterns': ['4WP'],
        'source_patterns': ['4 weeks of hiit', 'four weeks of hiit'],
        'characteristic_patterns': ['4 week', '4-week', 'four week']
    },
    '8w_hiit': {
        'title_patterns': ['8WP'],
        'source_patterns': ['8 weeks of hiit', 'eight weeks of hiit'],
        'characteristic_patterns': ['8 week', '8-week', 'eight week']
    },
    '12w_hiit': {
        'title_patterns': ['12WP'],
        'source_patterns': ['12 weeks of hiit', 'twelve weeks of hiit'],
        'characteristic_patterns': ['12 week', '12-week', 'twelve week']
    },
    'control': {
        'title_patterns': ['PO', 'CON'],
        'source_patterns': ['control period'],
        'characteristic_patterns': ['control']
    }
}

# Sample name parsing pattern
SAMPLE_NAME_PATTERN = r'Muscle_([A-Z0-9]+)(?:_Inter([12]))?_GS(\d+)_Rep(\d+)'


class SampleMapper:
    """
    A class for parsing and mapping sample metadata from GEO datasets.

    This class handles the extraction of experimental information from
    sample names and metadata, creating structured mappings for downstream
    analysis including binary classification, multiclass classification,
    and time-series trajectory analysis.

    Attributes:
        individual_id_pattern: Regex pattern for extracting individual IDs.
        timepoint_patterns: Dictionary of patterns for identifying timepoints.

    Example:
        >>> mapper = SampleMapper()
        >>> mapping = mapper.create_sample_mapping(metadata)
        >>> binary_labels = mapper.create_binary_labels(mapping)
    """

    def __init__(
        self,
        individual_id_pattern: str = r'GS\d+',
        timepoint_patterns: Optional[Dict] = None
    ) -> None:
        """
        Initialize the sample mapper.

        Args:
            individual_id_pattern: Regex pattern for extracting individual IDs
                from sample names.
            timepoint_patterns: Custom timepoint detection patterns.
                If None, uses default TIMEPOINT_PATTERNS.

        Example:
            >>> mapper = SampleMapper()
            >>> mapper = SampleMapper(individual_id_pattern=r'Subject\d+')
        """
        self.individual_id_pattern = individual_id_pattern
        self.timepoint_patterns = timepoint_patterns or TIMEPOINT_PATTERNS
        self._mapping_cache: Optional[pd.DataFrame] = None

    def parse_sample_title(
        self,
        title: str
    ) -> Dict[str, Any]:
        """
        Extract experimental information from a sample title.

        Parses structured sample names to extract timepoint, individual ID,
        intervention group, and replicate information.

        Args:
            title: Sample title string (e.g., 'Muscle_4WP_GS11_Rep1').

        Returns:
            Dictionary with keys:
                - 'time_code': Raw timepoint code (e.g., '4WP')
                - 'intervention_group': Intervention group (e.g., '1', '2', or None)
                - 'individual_id': Subject ID (e.g., 'GS11')
                - 'replicate': Replicate number
                - 'time_point': Standardized timepoint name
                - 'study_group': Study group classification

        Example:
            >>> mapper = SampleMapper()
            >>> info = mapper.parse_sample_title('Muscle_4WP_Inter2_GS11_Rep1')
            >>> print(info['time_point'])  # '4W HIIT'
            >>> print(info['individual_id'])  # 'GS11'
        """
        result = {
            'time_code': None,
            'intervention_group': None,
            'individual_id': '',
            'replicate': 1,
            'time_point': 'Unknown',
            'study_group': 'Main'
        }

        # Try structured pattern first
        match = re.search(SAMPLE_NAME_PATTERN, title)
        if match:
            time_code = match.group(1)
            intervention_group = match.group(2)
            individual_num = match.group(3)
            replicate = int(match.group(4))

            result['time_code'] = time_code
            result['intervention_group'] = intervention_group
            result['individual_id'] = f'GS{individual_num}'
            result['replicate'] = replicate

            # Map time code to standardized timepoint
            time_point_map = {
                'PRE': 'Baseline',
                '4WP': '4W HIIT',
                '8WP': '8W HIIT',
                '12WP': '12W HIIT',
                'PO': 'Control',
                'CON': 'Control-Inter'
            }
            result['time_point'] = time_point_map.get(time_code, 'Unknown')

            # Determine study group
            if intervention_group:
                result['study_group'] = f'Inter{intervention_group}'

        else:
            # Fallback: try to extract individual ID
            id_match = re.search(self.individual_id_pattern, title)
            if id_match:
                result['individual_id'] = id_match.group(0)

            # Try to determine timepoint from title
            title_upper = title.upper()
            for tp_name, patterns in self.timepoint_patterns.items():
                for pattern in patterns.get('title_patterns', []):
                    if pattern in title_upper:
                        result['time_point'] = self._normalize_timepoint(tp_name)
                        break

        return result

    def _normalize_timepoint(self, timepoint_key: str) -> str:
        """
        Convert internal timepoint key to display name.

        Args:
            timepoint_key: Internal key (e.g., '4w_hiit').

        Returns:
            Display name (e.g., '4W HIIT').
        """
        mapping = {
            'baseline': 'Baseline',
            '4w_hiit': '4W HIIT',
            '8w_hiit': '8W HIIT',
            '12w_hiit': '12W HIIT',
            'control': 'Control'
        }
        return mapping.get(timepoint_key, 'Unknown')

    def _extract_age_sex(
        self,
        characteristics: List[str],
        source: str = ''
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Extract age and sex from sample characteristics.

        Args:
            characteristics: List of characteristic strings.
            source: Source description string.

        Returns:
            Tuple of (age, sex) where age is float and sex is 'M' or 'F'.
        """
        age = None
        sex = None

        # Search in characteristics
        for char in characteristics:
            char_lower = char.lower()

            # Age extraction
            if 'age:' in char_lower and age is None:
                age_match = re.search(r'age:\s*([\d.]+)', char_lower)
                if age_match:
                    try:
                        age = float(age_match.group(1))
                    except ValueError:
                        pass

            # Sex extraction
            if 'sex:' in char_lower and sex is None:
                sex_match = re.search(r'sex:\s*([MFmf])', char_lower)
                if sex_match:
                    sex = sex_match.group(1).upper()

        # Fallback to source description
        source_lower = source.lower()

        if age is None and 'age:' in source_lower:
            age_match = re.search(r'age:\s*([\d.]+)', source_lower)
            if age_match:
                try:
                    age = float(age_match.group(1))
                except ValueError:
                    pass

        if sex is None and 'sex:' in source_lower:
            sex_match = re.search(r'sex:\s*([MFmf])', source_lower)
            if sex_match:
                sex = sex_match.group(1).upper()

        return age, sex

    def create_sample_mapping(
        self,
        metadata: Dict[str, List[str]],
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create a comprehensive sample mapping DataFrame from metadata.

        Parses all sample metadata to create a structured mapping table
        with experimental information, demographic data, and classification labels.

        Args:
            metadata: Dictionary of metadata from GEO series matrix.
                Expected keys include 'Sample_geo_accession', 'Sample_title',
                'Sample_source_name_ch1', 'Sample_characteristics_ch1'.
            output_path: Optional path to save the mapping as CSV.

        Returns:
            DataFrame with columns:
                - sample_id: GEO sample accession
                - sample_name: Original sample title
                - individual_id: Subject identifier
                - time_point: Experimental timepoint
                - age: Subject age
                - sex: Subject sex
                - binary_class: Binary classification label
                - multi_class: Multiclass label for HIIT duration
                - study_group: Study group assignment
                - is_duplicate: Flag for potential duplicates

        Example:
            >>> mapper = SampleMapper()
            >>> mapping = mapper.create_sample_mapping(metadata)
            >>> print(mapping['time_point'].value_counts())
        """
        logger.info("Creating sample mapping from metadata")

        # Extract required fields
        sample_ids = metadata.get('Sample_geo_accession', [])
        sample_titles = metadata.get('Sample_title', [])
        sample_sources = metadata.get('Sample_source_name_ch1', [])

        # Handle characteristics (may be concatenated with '; ')
        raw_characteristics = metadata.get('Sample_characteristics_ch1', [])
        sample_characteristics = {}
        for i, sample_id in enumerate(sample_ids):
            if i < len(raw_characteristics):
                char_str = raw_characteristics[i]
                sample_characteristics[sample_id] = char_str.split('; ')
            else:
                sample_characteristics[sample_id] = []

        logger.info(f"Processing {len(sample_ids)} samples")

        # Build mapping data
        mapping_data: Dict[str, List] = defaultdict(list)

        for i, sample_id in enumerate(sample_ids):
            title = sample_titles[i] if i < len(sample_titles) else ''
            source = sample_sources[i] if i < len(sample_sources) else ''
            chars = sample_characteristics.get(sample_id, [])

            # Parse sample title
            title_info = self.parse_sample_title(title)

            # Extract age and sex
            age, sex = self._extract_age_sex(chars, source)

            # Determine binary class
            time_point = title_info['time_point']
            if time_point in ['Baseline', 'Control', 'Control-Inter']:
                binary_class = 'Control'
            elif time_point == 'Unknown':
                binary_class = 'Unknown'
            else:
                binary_class = 'HIIT'

            # Determine multiclass label
            multi_class = np.nan
            if time_point == '4W HIIT':
                multi_class = '4W'
            elif time_point == '8W HIIT':
                multi_class = '8W'
            elif time_point == '12W HIIT':
                multi_class = '12W'

            # Add to mapping
            mapping_data['sample_id'].append(sample_id)
            mapping_data['sample_name'].append(title)
            mapping_data['source_description'].append(source)
            mapping_data['characteristics'].append('; '.join(chars))
            mapping_data['individual_id'].append(title_info['individual_id'])
            mapping_data['time_point'].append(time_point)
            mapping_data['age'].append(age)
            mapping_data['sex'].append(sex)
            mapping_data['binary_class'].append(binary_class)
            mapping_data['multi_class'].append(multi_class)
            mapping_data['study_group'].append(title_info['study_group'])
            mapping_data['intervention_group'].append(title_info['intervention_group'])
            mapping_data['replicate'].append(title_info['replicate'])

        # Create DataFrame
        mapping_df = pd.DataFrame(mapping_data)

        # Detect duplicates
        mapping_df['is_duplicate'] = mapping_df.duplicated(
            subset=['individual_id', 'time_point'],
            keep=False
        )

        # Log statistics
        self._log_mapping_statistics(mapping_df)

        # Cache the mapping
        self._mapping_cache = mapping_df

        # Save if path provided
        if output_path:
            mapping_df.to_csv(output_path, index=False)
            logger.info(f"Mapping saved to: {output_path}")

        return mapping_df

    def _log_mapping_statistics(self, mapping_df: pd.DataFrame) -> None:
        """Log statistics about the sample mapping."""
        logger.info(f"Total samples: {len(mapping_df)}")

        time_counts = mapping_df['time_point'].value_counts()
        logger.info("Timepoint distribution:")
        for tp, count in time_counts.items():
            logger.info(f"  {tp}: {count}")

        binary_counts = mapping_df['binary_class'].value_counts()
        logger.info("Binary class distribution:")
        for cls, count in binary_counts.items():
            logger.info(f"  {cls}: {count}")

        dup_count = mapping_df['is_duplicate'].sum()
        if dup_count > 0:
            logger.warning(f"Found {dup_count} potential duplicate samples")

    def create_binary_labels(
        self,
        mapping: pd.DataFrame,
        positive_class: str = 'HIIT',
        exclude_unknown: bool = True
    ) -> pd.Series:
        """
        Create binary classification labels.

        Args:
            mapping: Sample mapping DataFrame.
            positive_class: Label for positive class (default 'HIIT').
            exclude_unknown: If True, set Unknown samples to NaN.

        Returns:
            Series with binary labels (0 for Control, 1 for HIIT).

        Example:
            >>> labels = mapper.create_binary_labels(mapping)
            >>> print(labels.value_counts())
        """
        labels = mapping['binary_class'].copy()

        if exclude_unknown:
            labels = labels.replace('Unknown', np.nan)

        # Convert to numeric
        numeric_labels = labels.map({
            'Control': 0,
            positive_class: 1
        })

        return numeric_labels

    def create_multiclass_labels(
        self,
        mapping: pd.DataFrame,
        include_baseline: bool = False
    ) -> pd.Series:
        """
        Create multiclass labels for HIIT duration classification.

        Args:
            mapping: Sample mapping DataFrame.
            include_baseline: If True, include Baseline as class 0.

        Returns:
            Series with multiclass labels (NaN for non-HIIT samples if
            include_baseline is False).

        Example:
            >>> labels = mapper.create_multiclass_labels(mapping)
            >>> # Returns 0=4W, 1=8W, 2=12W for HIIT samples
        """
        if include_baseline:
            label_map = {
                'Baseline': 0,
                '4W': 1,
                '8W': 2,
                '12W': 3
            }
            # Use time_point for baseline, multi_class for HIIT
            labels = mapping.apply(
                lambda row: 0 if row['time_point'] == 'Baseline'
                else label_map.get(row['multi_class'], np.nan),
                axis=1
            )
        else:
            label_map = {'4W': 0, '8W': 1, '12W': 2}
            labels = mapping['multi_class'].map(label_map)

        return labels

    def create_timeseries_labels(
        self,
        mapping: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create labels for time-series trajectory analysis.

        Creates a structured label format for analyzing methylation
        changes over time within individuals.

        Args:
            mapping: Sample mapping DataFrame.

        Returns:
            DataFrame with columns:
                - sample_id: Sample identifier
                - individual_id: Subject identifier
                - time_numeric: Numeric time (0=Baseline, 4=4W, 8=8W, 12=12W)
                - time_point: Original timepoint label
                - is_hiit: Boolean indicating HIIT intervention

        Example:
            >>> ts_labels = mapper.create_timeseries_labels(mapping)
            >>> # Can be used to track individual trajectories
        """
        # Map timepoints to numeric values
        time_map = {
            'Baseline': 0,
            'Control': 0,
            'Control-Inter': 0,
            '4W HIIT': 4,
            '8W HIIT': 8,
            '12W HIIT': 12
        }

        ts_df = pd.DataFrame({
            'sample_id': mapping['sample_id'],
            'individual_id': mapping['individual_id'],
            'time_numeric': mapping['time_point'].map(time_map),
            'time_point': mapping['time_point'],
            'is_hiit': mapping['binary_class'] == 'HIIT'
        })

        # Filter out unknown timepoints
        ts_df = ts_df[ts_df['time_numeric'].notna()]

        return ts_df

    def detect_duplicates(
        self,
        mapping: pd.DataFrame,
        subset: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Detect and report duplicate samples in the mapping.

        Args:
            mapping: Sample mapping DataFrame.
            subset: Columns to consider for duplicate detection.
                Default is ['individual_id', 'time_point'].

        Returns:
            DataFrame containing only duplicate samples, sorted by
            individual_id and time_point.

        Example:
            >>> duplicates = mapper.detect_duplicates(mapping)
            >>> print(f"Found {len(duplicates)} duplicate samples")
        """
        if subset is None:
            subset = ['individual_id', 'time_point']

        # Find duplicates
        duplicate_mask = mapping.duplicated(subset=subset, keep=False)
        duplicates = mapping[duplicate_mask].sort_values(subset)

        if len(duplicates) > 0:
            logger.info(f"Found {len(duplicates)} duplicate samples")

            # Group and report
            for _, group in duplicates.groupby(subset):
                if len(group) > 1:
                    logger.debug(
                        f"Duplicate group: {group['sample_id'].tolist()}"
                    )

        return duplicates

    def get_samples_by_class(
        self,
        mapping: pd.DataFrame,
        class_column: str = 'binary_class',
        class_value: str = 'HIIT'
    ) -> List[str]:
        """
        Get sample IDs for a specific class.

        Args:
            mapping: Sample mapping DataFrame.
            class_column: Column containing class labels.
            class_value: Class value to filter for.

        Returns:
            List of sample IDs belonging to the specified class.

        Example:
            >>> hiit_samples = mapper.get_samples_by_class(mapping, 'binary_class', 'HIIT')
            >>> control_samples = mapper.get_samples_by_class(mapping, 'binary_class', 'Control')
        """
        mask = mapping[class_column] == class_value
        return mapping.loc[mask, 'sample_id'].tolist()


def parse_sample_metadata(
    metadata: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Parse sample metadata into a structured DataFrame.

    Convenience function for creating sample mappings without
    instantiating a SampleMapper object.

    Args:
        metadata: Dictionary of metadata from GEO series matrix.
            Expected keys include 'Sample_geo_accession', 'Sample_title'.

    Returns:
        DataFrame with parsed sample information.

    Example:
        >>> mapping = parse_sample_metadata(metadata)
        >>> print(mapping.columns.tolist())
    """
    mapper = SampleMapper()
    return mapper.create_sample_mapping(metadata)
