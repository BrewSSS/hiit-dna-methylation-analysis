"""
Database interface classes for enrichment analysis.

This module provides interfaces to various biological databases used in
enrichment analysis, including Gene Ontology, KEGG pathways, and MSigDB
gene sets.

Classes:
    GOAnnotator: Gene Ontology annotation interface
    KEGGPathwayMapper: KEGG pathway mapping interface
    MSigDBLoader: MSigDB gene set loader for fallback enrichment
    EPICAnnotationMapper: EPIC/450K array annotation mapper
"""

import json
import logging
import os
import re
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union

import pandas as pd

logger = logging.getLogger(__name__)


class GOAnnotator:
    """
    Gene Ontology annotation interface.

    Provides methods to query GO terms associated with genes, retrieve term
    descriptions, and navigate the GO term hierarchy. Supports both local
    OBO file parsing and remote API queries.

    Attributes:
        obo_file: Path to local GO OBO file (optional)
        terms: Dictionary mapping GO IDs to term information
        gene_annotations: Dictionary mapping genes to their GO annotations

    Example:
        >>> annotator = GOAnnotator()
        >>> terms = annotator.get_go_terms(['TP53', 'BRCA1'])
        >>> description = annotator.get_term_description('GO:0006915')
    """

    # GO namespaces
    ONTOLOGIES = {
        'BP': 'biological_process',
        'MF': 'molecular_function',
        'CC': 'cellular_component'
    }

    # QuickGO API base URL
    QUICKGO_API_URL = "https://www.ebi.ac.uk/QuickGO/services"

    def __init__(self, obo_file: Optional[str] = None):
        """
        Initialize the GO annotator.

        Args:
            obo_file: Optional path to a local GO OBO file. If not provided,
                     the annotator will use remote API queries.
        """
        self.obo_file = obo_file
        self.terms: Dict[str, Dict[str, Any]] = {}
        self.gene_annotations: Dict[str, List[str]] = {}
        self._loaded = False

        if obo_file and os.path.exists(obo_file):
            self._load_obo_file(obo_file)

    def _load_obo_file(self, obo_path: str) -> None:
        """
        Parse and load a GO OBO file.

        Args:
            obo_path: Path to the OBO file
        """
        try:
            with open(obo_path, 'r') as f:
                content = f.read()

            # Parse terms from OBO format
            term_blocks = re.split(r'\n\[Term\]\n', content)

            for block in term_blocks[1:]:  # Skip header
                lines = block.strip().split('\n')
                term_data: Dict[str, Any] = {}

                for line in lines:
                    if line.startswith('['):
                        break
                    if ': ' in line:
                        key, value = line.split(': ', 1)
                        if key == 'id':
                            term_data['id'] = value
                        elif key == 'name':
                            term_data['name'] = value
                        elif key == 'namespace':
                            term_data['namespace'] = value
                        elif key == 'def':
                            # Extract definition text
                            match = re.match(r'"([^"]*)"', value)
                            if match:
                                term_data['definition'] = match.group(1)
                        elif key == 'is_a':
                            if 'parents' not in term_data:
                                term_data['parents'] = []
                            parent_id = value.split(' ! ')[0]
                            term_data['parents'].append(parent_id)

                if 'id' in term_data:
                    self.terms[term_data['id']] = term_data

            self._loaded = True
            logger.info(f"Loaded {len(self.terms)} GO terms from {obo_path}")

        except Exception as e:
            logger.error(f"Failed to load OBO file: {e}")
            raise

    def get_go_terms(self, gene_list: List[str],
                     organism: str = 'hsapiens') -> Dict[str, List[Dict[str, Any]]]:
        """
        Get GO terms associated with a list of genes.

        Args:
            gene_list: List of gene symbols
            organism: Organism identifier (default: 'hsapiens' for human)

        Returns:
            Dictionary mapping gene symbols to lists of GO term annotations
        """
        results: Dict[str, List[Dict[str, Any]]] = {}

        for gene in gene_list:
            annotations = self._query_gene_annotations(gene, organism)
            if annotations:
                results[gene] = annotations

        return results

    def _query_gene_annotations(self, gene_symbol: str,
                                organism: str = 'hsapiens') -> List[Dict[str, Any]]:
        """
        Query GO annotations for a single gene using QuickGO API.

        Args:
            gene_symbol: Gene symbol to query
            organism: Organism identifier

        Returns:
            List of GO annotation dictionaries
        """
        # Check cache first
        if gene_symbol in self.gene_annotations:
            return [{'go_id': go_id} for go_id in self.gene_annotations[gene_symbol]]

        # Query QuickGO API
        try:
            params = {
                'geneProductId': gene_symbol,
                'taxonId': '9606' if organism == 'hsapiens' else organism,
                'limit': '100'
            }

            query_string = urllib.parse.urlencode(params)
            url = f"{self.QUICKGO_API_URL}/annotation/search?{query_string}"

            req = urllib.request.Request(url)
            req.add_header('Accept', 'application/json')

            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))

            annotations = []
            if 'results' in data:
                for ann in data['results']:
                    annotations.append({
                        'go_id': ann.get('goId'),
                        'go_name': ann.get('goName'),
                        'aspect': ann.get('goAspect'),
                        'evidence': ann.get('goEvidence'),
                        'qualifier': ann.get('qualifier')
                    })

            # Cache the results
            self.gene_annotations[gene_symbol] = [a['go_id'] for a in annotations if a['go_id']]

            return annotations

        except Exception as e:
            logger.warning(f"Failed to query GO annotations for {gene_symbol}: {e}")
            return []

    def get_term_description(self, go_id: str) -> Optional[str]:
        """
        Get the description for a GO term.

        Args:
            go_id: GO term identifier (e.g., 'GO:0006915')

        Returns:
            Term description string, or None if not found
        """
        # Check local cache first
        if go_id in self.terms:
            return self.terms[go_id].get('definition') or self.terms[go_id].get('name')

        # Query QuickGO API
        try:
            url = f"{self.QUICKGO_API_URL}/ontology/go/terms/{go_id}"

            req = urllib.request.Request(url)
            req.add_header('Accept', 'application/json')

            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))

            if 'results' in data and data['results']:
                term = data['results'][0]
                # Cache the term
                self.terms[go_id] = {
                    'id': go_id,
                    'name': term.get('name'),
                    'definition': term.get('definition', {}).get('text')
                }
                return term.get('definition', {}).get('text') or term.get('name')

            return None

        except Exception as e:
            logger.warning(f"Failed to get description for {go_id}: {e}")
            return None

    def get_term_hierarchy(self, go_id: str) -> Dict[str, List[str]]:
        """
        Get parent and child relationships for a GO term.

        Args:
            go_id: GO term identifier

        Returns:
            Dictionary with 'parents' and 'children' lists of GO IDs
        """
        hierarchy = {'parents': [], 'children': []}

        # Check local cache
        if go_id in self.terms and 'parents' in self.terms[go_id]:
            hierarchy['parents'] = self.terms[go_id]['parents']
            return hierarchy

        # Query QuickGO API for hierarchy
        try:
            url = f"{self.QUICKGO_API_URL}/ontology/go/terms/{go_id}/ancestors"

            req = urllib.request.Request(url)
            req.add_header('Accept', 'application/json')

            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))

            if 'results' in data and data['results']:
                # Extract direct parents
                for result in data['results']:
                    if result.get('relation') == 'is_a':
                        hierarchy['parents'].append(result.get('parent'))

            # Query children
            url = f"{self.QUICKGO_API_URL}/ontology/go/terms/{go_id}/children"

            req = urllib.request.Request(url)
            req.add_header('Accept', 'application/json')

            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))

            if 'results' in data and data['results']:
                for result in data['results']:
                    hierarchy['children'].append(result.get('child'))

            return hierarchy

        except Exception as e:
            logger.warning(f"Failed to get hierarchy for {go_id}: {e}")
            return hierarchy


class KEGGPathwayMapper:
    """
    KEGG pathway mapping interface.

    Provides methods to map genes to KEGG pathways, retrieve pathway
    information, and query genes within specific pathways.

    Attributes:
        organism: KEGG organism code (default: 'hsa' for human)
        pathways: Cached pathway information
        gene_pathway_map: Cached gene to pathway mappings

    Example:
        >>> mapper = KEGGPathwayMapper(organism='hsa')
        >>> pathways = mapper.map_genes_to_pathways(['TP53', 'BRCA1'])
        >>> genes = mapper.get_pathway_genes('hsa04110')
    """

    # KEGG REST API base URL
    KEGG_API_URL = "https://rest.kegg.jp"

    def __init__(self, organism: str = 'hsa'):
        """
        Initialize the KEGG pathway mapper.

        Args:
            organism: KEGG organism code (default: 'hsa' for Homo sapiens)
        """
        self.organism = organism
        self.pathways: Dict[str, Dict[str, Any]] = {}
        self.gene_pathway_map: Dict[str, List[str]] = {}
        self._pathway_genes: Dict[str, Set[str]] = {}

    def get_pathway_genes(self, pathway_id: str) -> List[str]:
        """
        Get all genes in a KEGG pathway.

        Args:
            pathway_id: KEGG pathway identifier (e.g., 'hsa04110')

        Returns:
            List of gene symbols in the pathway
        """
        # Check cache
        if pathway_id in self._pathway_genes:
            return list(self._pathway_genes[pathway_id])

        try:
            # Query KEGG API for pathway genes
            url = f"{self.KEGG_API_URL}/get/{pathway_id}"

            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read().decode('utf-8')

            # Parse genes from pathway entry
            genes = set()
            in_gene_section = False

            for line in content.split('\n'):
                if line.startswith('GENE'):
                    in_gene_section = True
                    # Extract gene from first GENE line
                    parts = line.replace('GENE', '').strip().split()
                    if len(parts) >= 2:
                        # Gene symbol is typically the second element
                        gene_info = parts[1].rstrip(';')
                        genes.add(gene_info)
                elif in_gene_section:
                    if line.startswith(' ') or line.startswith('\t'):
                        # Continuation of GENE section
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            gene_info = parts[1].rstrip(';')
                            genes.add(gene_info)
                    else:
                        # End of GENE section
                        in_gene_section = False

            self._pathway_genes[pathway_id] = genes
            return list(genes)

        except Exception as e:
            logger.warning(f"Failed to get genes for pathway {pathway_id}: {e}")
            return []

    def map_genes_to_pathways(self, gene_list: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Map a list of genes to their KEGG pathways.

        Args:
            gene_list: List of gene symbols

        Returns:
            Dictionary mapping gene symbols to lists of pathway information
        """
        results: Dict[str, List[Dict[str, Any]]] = {}

        for gene in gene_list:
            pathways = self._query_gene_pathways(gene)
            if pathways:
                results[gene] = pathways

        return results

    def _query_gene_pathways(self, gene_symbol: str) -> List[Dict[str, Any]]:
        """
        Query KEGG pathways for a single gene.

        Args:
            gene_symbol: Gene symbol to query

        Returns:
            List of pathway dictionaries
        """
        # Check cache
        if gene_symbol in self.gene_pathway_map:
            return [{'pathway_id': pid} for pid in self.gene_pathway_map[gene_symbol]]

        try:
            # First, find the KEGG gene ID
            url = f"{self.KEGG_API_URL}/find/{self.organism}/{gene_symbol}"

            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read().decode('utf-8')

            if not content.strip():
                return []

            # Parse KEGG gene ID from first result
            first_line = content.strip().split('\n')[0]
            kegg_gene_id = first_line.split('\t')[0]

            # Query pathways for this gene
            url = f"{self.KEGG_API_URL}/link/pathway/{kegg_gene_id}"

            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read().decode('utf-8')

            pathways = []
            for line in content.strip().split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        pathway_id = parts[1].replace('path:', '')
                        pathways.append({
                            'pathway_id': pathway_id,
                            'gene_id': kegg_gene_id
                        })

            # Cache the results
            self.gene_pathway_map[gene_symbol] = [p['pathway_id'] for p in pathways]

            return pathways

        except Exception as e:
            logger.warning(f"Failed to query pathways for {gene_symbol}: {e}")
            return []

    def get_pathway_description(self, pathway_id: str) -> Optional[str]:
        """
        Get the description/name of a KEGG pathway.

        Args:
            pathway_id: KEGG pathway identifier

        Returns:
            Pathway description string, or None if not found
        """
        # Check cache
        if pathway_id in self.pathways:
            return self.pathways[pathway_id].get('name')

        try:
            url = f"{self.KEGG_API_URL}/get/{pathway_id}"

            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read().decode('utf-8')

            # Parse pathway name from entry
            for line in content.split('\n'):
                if line.startswith('NAME'):
                    name = line.replace('NAME', '').strip()
                    # Remove organism suffix if present
                    name = name.split(' - ')[0].strip()

                    # Cache the result
                    self.pathways[pathway_id] = {'id': pathway_id, 'name': name}
                    return name

            return None

        except Exception as e:
            logger.warning(f"Failed to get description for {pathway_id}: {e}")
            return None

    def get_all_pathways(self) -> List[Dict[str, str]]:
        """
        Get all available pathways for the organism.

        Returns:
            List of pathway dictionaries with 'id' and 'name' keys
        """
        try:
            url = f"{self.KEGG_API_URL}/list/pathway/{self.organism}"

            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read().decode('utf-8')

            pathways = []
            for line in content.strip().split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        pathway_id = parts[0].replace('path:', '')
                        name = parts[1].split(' - ')[0].strip()
                        pathways.append({'id': pathway_id, 'name': name})
                        self.pathways[pathway_id] = {'id': pathway_id, 'name': name}

            return pathways

        except Exception as e:
            logger.warning(f"Failed to get pathway list: {e}")
            return []


class MSigDBLoader:
    """
    MSigDB gene set loader for fallback enrichment analysis.

    Loads and manages gene sets from the Molecular Signatures Database (MSigDB)
    in GMT format. This is the core component of the fallback enrichment
    methodology, enabling hypergeometric enrichment testing without requiring
    R or rpy2 dependencies.

    Attributes:
        gmt_dir: Directory containing GMT files
        gene_sets: Dictionary mapping gene set names to gene sets
        collections_loaded: Set of loaded collection identifiers

    Example:
        >>> loader = MSigDBLoader(gmt_dir='/path/to/msigdb/')
        >>> loader.load_gene_sets(collections=['H', 'C2', 'C5'])
        >>> gene_set = loader.get_gene_set('HALLMARK_APOPTOSIS')
        >>> results = loader.search_gene_sets('metabolism')
    """

    # MSigDB collection descriptions
    COLLECTIONS = {
        'H': 'hallmark',
        'C1': 'positional',
        'C2': 'curated',
        'C3': 'motif',
        'C4': 'computational',
        'C5': 'GO',
        'C6': 'oncogenic',
        'C7': 'immunologic',
        'C8': 'cell_type'
    }

    # MSigDB subcollection mappings
    SUBCOLLECTIONS = {
        'C2': ['CGP', 'CP', 'CP:BIOCARTA', 'CP:KEGG', 'CP:PID', 'CP:REACTOME', 'CP:WIKIPATHWAYS'],
        'C5': ['GO:BP', 'GO:CC', 'GO:MF', 'HPO'],
        'C3': ['MIR:MIRDB', 'MIR:MIR_Legacy', 'TFT:GTRD', 'TFT:TFT_Legacy']
    }

    def __init__(self, gmt_dir: Optional[str] = None):
        """
        Initialize the MSigDB loader.

        Args:
            gmt_dir: Directory containing GMT files. If None, gene sets must
                    be loaded manually via parse_gmt_file().
        """
        self.gmt_dir = Path(gmt_dir) if gmt_dir else None
        self.gene_sets: Dict[str, Set[str]] = {}
        self.gene_set_metadata: Dict[str, Dict[str, Any]] = {}
        self.collections_loaded: Set[str] = set()

    def load_gene_sets(self, collections: Optional[List[str]] = None) -> int:
        """
        Load gene sets from MSigDB GMT files.

        Args:
            collections: List of collection identifiers to load (e.g., ['H', 'C2', 'C5']).
                        If None, loads hallmark (H), curated (C2), and GO (C5).

        Returns:
            Number of gene sets loaded

        Raises:
            ValueError: If gmt_dir is not set
            FileNotFoundError: If GMT files cannot be found
        """
        if self.gmt_dir is None:
            raise ValueError("gmt_dir must be set to load gene sets automatically")

        if collections is None:
            collections = ['H', 'C2', 'C5']

        loaded_count = 0

        for collection in collections:
            if collection not in self.COLLECTIONS:
                logger.warning(f"Unknown collection: {collection}")
                continue

            # Try to find GMT file for this collection
            collection_name = self.COLLECTIONS[collection]
            possible_files = [
                self.gmt_dir / f"{collection.lower()}.{collection_name}.symbols.gmt",
                self.gmt_dir / f"msigdb.{collection.lower()}.symbols.gmt",
                self.gmt_dir / f"{collection}.symbols.gmt",
                self.gmt_dir / f"{collection_name}.gmt"
            ]

            gmt_file = None
            for pf in possible_files:
                if pf.exists():
                    gmt_file = pf
                    break

            if gmt_file:
                count = self.parse_gmt_file(str(gmt_file))
                loaded_count += count
                self.collections_loaded.add(collection)
                logger.info(f"Loaded {count} gene sets from {collection}")
            else:
                logger.warning(f"GMT file not found for collection {collection}")

        return loaded_count

    def parse_gmt_file(self, gmt_path: str) -> int:
        """
        Parse a GMT format gene set file.

        GMT format: Each line contains:
        - Gene set name (tab-separated)
        - Description/URL (tab-separated)
        - Gene symbols (tab-separated)

        Args:
            gmt_path: Path to the GMT file

        Returns:
            Number of gene sets loaded from the file

        Raises:
            FileNotFoundError: If the file does not exist
        """
        if not os.path.exists(gmt_path):
            raise FileNotFoundError(f"GMT file not found: {gmt_path}")

        count = 0

        try:
            with open(gmt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split('\t')
                    if len(parts) < 3:
                        continue

                    set_name = parts[0]
                    description = parts[1]
                    genes = set(parts[2:])

                    # Store gene set
                    self.gene_sets[set_name] = genes
                    self.gene_set_metadata[set_name] = {
                        'name': set_name,
                        'description': description,
                        'size': len(genes),
                        'source_file': gmt_path
                    }
                    count += 1

            logger.info(f"Parsed {count} gene sets from {gmt_path}")

        except Exception as e:
            logger.error(f"Error parsing GMT file {gmt_path}: {e}")
            raise

        return count

    def get_gene_set(self, set_name: str) -> Optional[Set[str]]:
        """
        Get genes in a specific gene set.

        Args:
            set_name: Name of the gene set

        Returns:
            Set of gene symbols, or None if gene set not found
        """
        return self.gene_sets.get(set_name)

    def get_gene_set_info(self, set_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a gene set.

        Args:
            set_name: Name of the gene set

        Returns:
            Dictionary with gene set metadata, or None if not found
        """
        return self.gene_set_metadata.get(set_name)

    def search_gene_sets(self, keyword: str,
                         case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """
        Search gene sets by keyword in name or description.

        Args:
            keyword: Search keyword
            case_sensitive: Whether search should be case-sensitive

        Returns:
            List of matching gene set metadata dictionaries
        """
        results = []
        search_term = keyword if case_sensitive else keyword.lower()

        for set_name, metadata in self.gene_set_metadata.items():
            name = set_name if case_sensitive else set_name.lower()
            desc = metadata.get('description', '')
            desc = desc if case_sensitive else desc.lower()

            if search_term in name or search_term in desc:
                results.append({
                    **metadata,
                    'genes': self.gene_sets.get(set_name, set())
                })

        return results

    def get_all_gene_sets(self) -> Dict[str, Set[str]]:
        """
        Get all loaded gene sets.

        Returns:
            Dictionary mapping gene set names to gene sets
        """
        return self.gene_sets.copy()

    def get_collection_gene_sets(self, collection: str) -> Dict[str, Set[str]]:
        """
        Get gene sets from a specific collection.

        Args:
            collection: Collection identifier (e.g., 'H', 'C2')

        Returns:
            Dictionary of gene sets from the specified collection
        """
        prefix_map = {
            'H': 'HALLMARK_',
            'C2': ['BIOCARTA_', 'KEGG_', 'PID_', 'REACTOME_', 'WP_'],
            'C5': ['GO_', 'GOBP_', 'GOCC_', 'GOMF_'],
            'C6': 'ONCOGENIC_',
            'C7': 'IMMUNESIGDB_',
            'C8': 'CELLTYPE_'
        }

        prefixes = prefix_map.get(collection, [])
        if isinstance(prefixes, str):
            prefixes = [prefixes]

        result = {}
        for set_name, genes in self.gene_sets.items():
            if any(set_name.startswith(p) for p in prefixes):
                result[set_name] = genes

        return result

    def create_background_set(self) -> Set[str]:
        """
        Create a background gene set from all loaded gene sets.

        Returns:
            Set of all unique gene symbols across all gene sets
        """
        background = set()
        for genes in self.gene_sets.values():
            background.update(genes)
        return background

    def summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of loaded gene sets.

        Returns:
            Dictionary with summary statistics
        """
        sizes = [len(genes) for genes in self.gene_sets.values()]

        return {
            'total_gene_sets': len(self.gene_sets),
            'collections_loaded': list(self.collections_loaded),
            'total_unique_genes': len(self.create_background_set()),
            'min_set_size': min(sizes) if sizes else 0,
            'max_set_size': max(sizes) if sizes else 0,
            'mean_set_size': sum(sizes) / len(sizes) if sizes else 0
        }


class EPICAnnotationMapper:
    """
    EPIC array annotation mapper for CpG-to-gene mapping.

    Maps CpG probe IDs from Illumina MethylationEPIC or HumanMethylation450 arrays
    to gene symbols and genomic features using the array manifest annotation files.

    Attributes:
        annotation_file: Path to the annotation file
        annotations: DataFrame containing loaded annotations
        platform: Detected array platform ('EPIC', '450K', or 'unknown')

    Example:
        >>> mapper = EPICAnnotationMapper('GPL21145_MethylationEPIC_manifest.csv')
        >>> genes_map = mapper.map_to_genes(['cg00000029', 'cg00000165'])
        >>> context = mapper.get_genomic_context('cg00000029')
    """

    # Required columns for annotation mapping
    REQUIRED_COLUMNS = ['Name', 'UCSC_RefGene_Name']

    # Alternative column names for different annotation formats
    COLUMN_ALIASES = {
        'Name': ['IlmnID', 'Probe_ID', 'ProbeID', 'cpg_id', 'CpG_ID'],
        'UCSC_RefGene_Name': ['Gene_Symbol', 'RefGene_Name', 'gene_symbol', 'Gene'],
        'UCSC_RefGene_Group': ['Gene_Region', 'RefGene_Group', 'region'],
        'Relation_to_UCSC_CpG_Island': ['CpG_Island_Relation', 'CGI_Relation', 'relation'],
        'CHR': ['Chromosome', 'chr', 'chromosome'],
        'MAPINFO': ['Position', 'position', 'pos']
    }

    # Genomic region classifications
    PROMOTER_REGIONS = ['TSS200', 'TSS1500', '1stExon', '5\'UTR']
    GENE_BODY_REGIONS = ['Body', 'ExonBnd', '3\'UTR']

    def __init__(self, annotation_file: Optional[str] = None):
        """
        Initialize the EPIC annotation mapper.

        Args:
            annotation_file: Path to the annotation CSV file. If provided,
                           annotations are loaded immediately.
        """
        self.annotation_file = annotation_file
        self.annotations: Optional[pd.DataFrame] = None
        self.platform: str = 'unknown'
        self._cpg_index: Optional[Dict[str, int]] = None

        if annotation_file:
            self.load(annotation_file)

    def load(self, annotation_file: str) -> None:
        """
        Load EPIC annotation file.

        Attempts multiple loading strategies to handle different annotation
        file formats (with or without header rows, different delimiters).

        Args:
            annotation_file: Path to the annotation CSV file

        Raises:
            FileNotFoundError: If annotation file does not exist
            ValueError: If required columns cannot be found
        """
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

        # Try different loading strategies
        loading_attempts = [
            {'skiprows': 0, 'sep': ','},
            {'skiprows': 7, 'sep': ','},
            {'skiprows': 8, 'sep': ','},
            {'skiprows': 0, 'sep': '\t'},
        ]

        df = None
        for params in loading_attempts:
            try:
                df = pd.read_csv(
                    annotation_file,
                    **params,
                    low_memory=False,
                    on_bad_lines='skip'
                )

                # Check for required columns or aliases
                if self._validate_columns(df):
                    break
                df = None

            except Exception:
                continue

        if df is None:
            raise ValueError(
                f"Could not load annotation file {annotation_file}. "
                f"Required columns not found: {self.REQUIRED_COLUMNS}"
            )

        # Standardize column names
        df = self._standardize_columns(df)

        self.annotations = df
        self.annotation_file = annotation_file
        self._build_cpg_index()
        self._detect_platform()

        logger.info(
            f"Loaded {len(df)} annotations from {annotation_file} "
            f"(platform: {self.platform})"
        )

    def _validate_columns(self, df: pd.DataFrame) -> bool:
        """
        Check if DataFrame has required columns or their aliases.

        Args:
            df: DataFrame to validate

        Returns:
            True if required columns found, False otherwise
        """
        for required_col in self.REQUIRED_COLUMNS:
            found = required_col in df.columns
            if not found:
                # Check aliases
                aliases = self.COLUMN_ALIASES.get(required_col, [])
                found = any(alias in df.columns for alias in aliases)
            if not found:
                return False
        return True

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to expected format.

        Args:
            df: DataFrame with potentially non-standard column names

        Returns:
            DataFrame with standardized column names
        """
        rename_map = {}

        for standard_name, aliases in self.COLUMN_ALIASES.items():
            if standard_name not in df.columns:
                for alias in aliases:
                    if alias in df.columns:
                        rename_map[alias] = standard_name
                        break

        if rename_map:
            df = df.rename(columns=rename_map)

        return df

    def _build_cpg_index(self) -> None:
        """Build index for fast CpG lookup."""
        if self.annotations is not None and 'Name' in self.annotations.columns:
            self._cpg_index = {
                cpg: idx
                for idx, cpg in enumerate(self.annotations['Name'].values)
            }

    def _detect_platform(self) -> None:
        """Detect array platform from annotation data."""
        if self.annotations is None:
            return

        n_probes = len(self.annotations)

        if n_probes > 800000:
            self.platform = 'EPIC'
        elif n_probes > 400000:
            self.platform = '450K'
        elif n_probes > 20000:
            self.platform = '27K'
        else:
            self.platform = 'unknown'

    def map_to_genes(
        self,
        cpg_ids: List[str],
        mapping_type: str = 'UCSC_RefGene_Name',
        unique: bool = True
    ) -> Dict[str, List[str]]:
        """
        Map CpG IDs to gene symbols.

        Args:
            cpg_ids: List of CpG probe identifiers
            mapping_type: Column to use for gene mapping.
                Options: 'UCSC_RefGene_Name', 'nearest', 'all'
            unique: If True, return unique genes per CpG (default: True)

        Returns:
            Dictionary mapping CpG IDs to lists of gene symbols
        """
        if self.annotations is None:
            logger.warning("Annotations not loaded")
            return {}

        if mapping_type not in ['UCSC_RefGene_Name', 'nearest', 'all']:
            mapping_type = 'UCSC_RefGene_Name'

        gene_col = 'UCSC_RefGene_Name'

        result: Dict[str, List[str]] = {}

        for cpg_id in cpg_ids:
            if self._cpg_index is not None and cpg_id in self._cpg_index:
                idx = self._cpg_index[cpg_id]
                gene_str = self.annotations.iloc[idx].get(gene_col, '')
            else:
                # Fallback to filter query
                mask = self.annotations['Name'] == cpg_id
                if mask.sum() == 0:
                    result[cpg_id] = []
                    continue
                gene_str = self.annotations.loc[mask, gene_col].iloc[0]

            if pd.isna(gene_str) or not str(gene_str).strip():
                result[cpg_id] = []
            else:
                # Parse semicolon-separated gene names
                genes = [g.strip() for g in str(gene_str).split(';') if g.strip()]
                if unique:
                    genes = list(dict.fromkeys(genes))  # Preserve order, remove duplicates
                result[cpg_id] = genes

        return result

    def get_all_genes(self, cpg_ids: List[str]) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Get all unique genes mapped from CpG list.

        Convenience method that returns both unique gene list and mapping dict.

        Args:
            cpg_ids: List of CpG probe identifiers

        Returns:
            Tuple of:
                - Sorted list of unique gene symbols
                - Dictionary mapping CpG IDs to gene lists
        """
        cpg_gene_map = self.map_to_genes(cpg_ids)

        all_genes: Set[str] = set()
        for genes in cpg_gene_map.values():
            all_genes.update(genes)

        return sorted(list(all_genes)), cpg_gene_map

    def get_genomic_context(self, cpg_id: str) -> Dict[str, Any]:
        """
        Get genomic context information for a CpG site.

        Returns information about chromosome, position, gene associations,
        and CpG island relationships.

        Args:
            cpg_id: CpG probe identifier

        Returns:
            Dictionary with genomic context information:
                - chromosome: Chromosome name
                - position: Genomic position
                - genes: Associated genes
                - gene_region: Gene region classification
                - cpg_island_relation: Relationship to CpG island
        """
        if self.annotations is None:
            return {}

        if self._cpg_index is not None and cpg_id in self._cpg_index:
            idx = self._cpg_index[cpg_id]
            row = self.annotations.iloc[idx]
        else:
            mask = self.annotations['Name'] == cpg_id
            if mask.sum() == 0:
                return {}
            row = self.annotations.loc[mask].iloc[0]

        context = {
            'cpg_id': cpg_id,
            'chromosome': row.get('CHR', ''),
            'position': row.get('MAPINFO', ''),
            'genes': row.get('UCSC_RefGene_Name', ''),
            'gene_region': row.get('UCSC_RefGene_Group', ''),
            'cpg_island_relation': row.get('Relation_to_UCSC_CpG_Island', ''),
            'strand': row.get('Strand', '')
        }

        # Clean up any NaN values
        for key, value in context.items():
            if pd.isna(value):
                context[key] = ''

        return context

    def filter_by_region(
        self,
        cpg_ids: List[str],
        regions: List[str] = None
    ) -> List[str]:
        """
        Filter CpGs by genomic region.

        Args:
            cpg_ids: List of CpG probe identifiers
            regions: List of regions to include. Defaults to promoter regions
                ['TSS200', 'TSS1500', '1stExon', '5\'UTR']

        Returns:
            Filtered list of CpG IDs in specified regions
        """
        if self.annotations is None:
            return []

        if regions is None:
            regions = self.PROMOTER_REGIONS

        filtered = []

        for cpg_id in cpg_ids:
            context = self.get_genomic_context(cpg_id)
            region_str = context.get('gene_region', '')

            if not region_str:
                continue

            # Check if any specified region is in the CpG's region annotation
            cpg_regions = [r.strip() for r in str(region_str).split(';')]
            if any(r in regions for r in cpg_regions):
                filtered.append(cpg_id)

        return filtered

    def get_promoter_cpgs(self, cpg_ids: List[str]) -> List[str]:
        """
        Filter to keep only CpGs in promoter regions.

        Args:
            cpg_ids: List of CpG probe identifiers

        Returns:
            List of CpG IDs located in promoter regions
        """
        return self.filter_by_region(cpg_ids, regions=self.PROMOTER_REGIONS)

    def get_gene_body_cpgs(self, cpg_ids: List[str]) -> List[str]:
        """
        Filter to keep only CpGs in gene body regions.

        Args:
            cpg_ids: List of CpG probe identifiers

        Returns:
            List of CpG IDs located in gene body regions
        """
        return self.filter_by_region(cpg_ids, regions=self.GENE_BODY_REGIONS)

    def summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of loaded annotations.

        Returns:
            Dictionary with summary statistics
        """
        if self.annotations is None:
            return {'loaded': False}

        return {
            'loaded': True,
            'platform': self.platform,
            'total_probes': len(self.annotations),
            'annotation_file': self.annotation_file,
            'columns_available': list(self.annotations.columns)
        }
