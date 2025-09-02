"""Protocol data processor with Mamba feature integration.

This module handles:
- Loading and preprocessing protocol packet datasets
- Extracting Mamba features from protocol data
- Managing protocol field structures and relationships
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
import struct
import random


class ProtocolDataProcessor:
    """Processes protocol datasets and extracts features for fuzzing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize protocol data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.mamba_feature_dim = config.get('mamba_feature_dim', 256)
        self.protocol_type = config.get('protocol_type', 'generic')
        self.dataset_path = config.get('dataset_path', None)
        
        # Protocol field definitions
        self.protocol_fields = config.get('protocol_fields', [])
        self.field_types = config.get('field_types', {})
        self.field_constraints = config.get('field_constraints', {})
        
        # Cached dataset
        self.dataset = []
        self.current_dataset_idx = 0
        
        # Feature extraction cache
        self.feature_cache = {}
        
        # Mamba model (placeholder - would be actual model in practice)
        self.mamba_model = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Load dataset if path provided
        if self.dataset_path:
            self.load_dataset(self.dataset_path)
            
    def load_dataset(self, dataset_path: str):
        """Load preprocessed protocol dataset.
        
        Args:
            dataset_path: Path to protocol dataset file
        """
        try:
            with open(dataset_path, 'r') as f:
                if dataset_path.endswith('.json'):
                    self.dataset = json.load(f)
                else:
                    # Handle other formats as needed
                    self.dataset = []
                    
            self.logger.info(f"Loaded {len(self.dataset)} protocol samples from {dataset_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load dataset from {dataset_path}: {e}")
            # Generate synthetic dataset for testing
            self.dataset = self._generate_synthetic_dataset()
            
    def _generate_synthetic_dataset(self, n_samples: int = 1000) -> List[Dict]:
        """Generate synthetic protocol dataset for testing.
        
        Args:
            n_samples: Number of synthetic samples to generate
            
        Returns:
            List of synthetic protocol packets
        """
        synthetic_data = []
        
        # Define some example protocol fields
        if not self.protocol_fields:
            self.protocol_fields = [
                'header', 'source_addr', 'dest_addr', 'payload', 
                'checksum', 'flags', 'sequence_num', 'timestamp'
            ]
            
        for i in range(n_samples):
            packet = {}
            
            for field in self.protocol_fields:
                # Generate synthetic field data based on field type
                if field in ['source_addr', 'dest_addr']:
                    # Generate IP-like addresses
                    packet[field] = struct.pack('>I', random.randint(0, 2**32 - 1))
                elif field == 'header':
                    # Generate header data
                    packet[field] = struct.pack('>HH', random.randint(0, 65535), random.randint(0, 65535))
                elif field == 'payload':
                    # Generate variable length payload
                    length = random.randint(10, 500)
                    packet[field] = bytes([random.randint(0, 255) for _ in range(length)])
                elif field in ['checksum', 'sequence_num', 'timestamp']:
                    # Generate 4-byte integers
                    packet[field] = struct.pack('>I', random.randint(0, 2**32 - 1))
                elif field == 'flags':
                    # Generate 1-byte flags
                    packet[field] = struct.pack('>B', random.randint(0, 255))
                else:
                    # Default: generate random bytes
                    length = random.randint(4, 32)
                    packet[field] = bytes([random.randint(0, 255) for _ in range(length)])
                    
            # Add metadata
            packet['_metadata'] = {
                'sample_id': i,
                'protocol_type': self.protocol_type,
                'total_size': sum(len(v) if isinstance(v, bytes) else 0 for k, v in packet.items() if not k.startswith('_'))
            }
            
            synthetic_data.append(packet)
            
        self.logger.info(f"Generated {n_samples} synthetic protocol samples")
        return synthetic_data
        
    def sample_protocol_data(self) -> Dict:
        """Sample a protocol packet from the dataset.
        
        Returns:
            Protocol packet dictionary
        """
        if not self.dataset:
            # Generate single synthetic packet if no dataset
            return self._generate_synthetic_dataset(1)[0]
            
        # Cycle through dataset
        packet = self.dataset[self.current_dataset_idx]
        self.current_dataset_idx = (self.current_dataset_idx + 1) % len(self.dataset)
        
        return packet.copy()  # Return copy to avoid modifying original
        
    def extract_mamba_features(self, protocol_data: Dict) -> np.ndarray:
        """Extract Mamba features from protocol data.
        
        This is a placeholder implementation. In practice, this would:
        1. Convert protocol data to sequence format suitable for Mamba
        2. Run through pre-trained Mamba model
        3. Return feature embeddings
        
        Args:
            protocol_data: Protocol packet dictionary
            
        Returns:
            Mamba feature vector of shape (mamba_feature_dim,)
        """
        # Create cache key from packet content
        cache_key = self._create_cache_key(protocol_data)
        
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
            
        # Placeholder Mamba feature extraction
        features = self._extract_features_placeholder(protocol_data)
        
        # Cache the result
        self.feature_cache[cache_key] = features
        
        return features
        
    def _create_cache_key(self, protocol_data: Dict) -> str:
        """Create cache key for protocol data."""
        # Simple hash based on packet content
        content = []
        for field in sorted(protocol_data.keys()):
            if not field.startswith('_'):  # Skip metadata
                value = protocol_data[field]
                if isinstance(value, bytes):
                    content.append(f"{field}:{len(value)}:{hash(value)}")
                else:
                    content.append(f"{field}:{value}")
        return '|'.join(content)
        
    def _extract_features_placeholder(self, protocol_data: Dict) -> np.ndarray:
        """Placeholder implementation of Mamba feature extraction.
        
        In practice, this would:
        1. Convert protocol fields to sequence representation
        2. Apply Mamba model for feature extraction
        3. Return contextual embeddings
        
        For now, creates features based on statistical properties of the data.
        """
        features = np.zeros(self.mamba_feature_dim)
        
        # Extract basic statistical features from each field
        field_features = []
        
        for i, field in enumerate(self.protocol_fields):
            if field in protocol_data:
                field_data = protocol_data[field]
                
                if isinstance(field_data, bytes):
                    # Extract byte-level statistics
                    if len(field_data) > 0:
                        field_stats = [
                            len(field_data),  # Length
                            np.mean(list(field_data)),  # Mean byte value
                            np.std(list(field_data)),   # Std byte value
                            np.min(list(field_data)),   # Min byte value
                            np.max(list(field_data)),   # Max byte value
                            len(set(field_data)),       # Unique bytes
                        ]
                    else:
                        field_stats = [0] * 6
                else:
                    # Handle non-bytes data
                    field_stats = [1, 0, 0, 0, 0, 1]  # Default stats
                    
                field_features.extend(field_stats)
            else:
                # Field not present
                field_features.extend([0] * 6)
                
        # Pad or truncate to desired feature dimension
        field_features = field_features[:self.mamba_feature_dim]
        field_features.extend([0] * (self.mamba_feature_dim - len(field_features)))
        
        features[:len(field_features)] = field_features
        
        # Add some random context to simulate Mamba's contextual understanding
        np.random.seed(hash(self._create_cache_key(protocol_data)) % 2**32)
        context_noise = np.random.normal(0, 0.1, self.mamba_feature_dim)
        features = features + context_noise
        
        # Normalize
        if np.linalg.norm(features) > 0:
            features = features / np.linalg.norm(features)
            
        return features
        
    def get_field_relationships(self) -> Dict[str, List[str]]:
        """Get relationships between protocol fields.
        
        Returns:
            Dictionary mapping each field to related fields
        """
        relationships = {}
        
        # Define some basic relationships for common protocol fields
        common_relationships = {
            'header': ['flags', 'checksum'],
            'source_addr': ['dest_addr'],
            'dest_addr': ['source_addr'],
            'payload': ['checksum'],
            'sequence_num': ['timestamp'],
            'checksum': ['header', 'payload'],
            'flags': ['header']
        }
        
        for field in self.protocol_fields:
            relationships[field] = common_relationships.get(field, [])
            
        return relationships
        
    def get_field_context(self, field_name: str, protocol_data: Dict) -> Dict[str, Any]:
        """Get contextual information for a specific field.
        
        Args:
            field_name: Name of the field
            protocol_data: Protocol packet data
            
        Returns:
            Context dictionary with field information
        """
        context = {
            'field_name': field_name,
            'field_type': self.field_types.get(field_name, 'bytes'),
            'field_present': field_name in protocol_data,
            'field_size': 0,
            'field_constraints': self.field_constraints.get(field_name, {}),
            'related_fields': []
        }
        
        if field_name in protocol_data:
            field_data = protocol_data[field_name]
            if isinstance(field_data, bytes):
                context['field_size'] = len(field_data)
            else:
                context['field_size'] = 1
                
        # Get related fields
        relationships = self.get_field_relationships()
        context['related_fields'] = relationships.get(field_name, [])
        
        return context
        
    def validate_packet(self, protocol_data: Dict) -> Dict[str, Any]:
        """Validate protocol packet structure and constraints.
        
        Args:
            protocol_data: Protocol packet data
            
        Returns:
            Validation result dictionary
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'missing_fields': [],
            'constraint_violations': []
        }
        
        # Check required fields
        for field in self.protocol_fields:
            if field not in protocol_data:
                validation['missing_fields'].append(field)
                validation['warnings'].append(f"Missing field: {field}")
                
        # Check field constraints
        for field, constraints in self.field_constraints.items():
            if field in protocol_data:
                field_data = protocol_data[field]
                
                # Size constraints
                if 'max_size' in constraints:
                    if isinstance(field_data, bytes) and len(field_data) > constraints['max_size']:
                        validation['constraint_violations'].append(f"{field} exceeds max size")
                        
                if 'min_size' in constraints:
                    if isinstance(field_data, bytes) and len(field_data) < constraints['min_size']:
                        validation['constraint_violations'].append(f"{field} below min size")
                        
        # Set overall validity
        validation['valid'] = len(validation['errors']) == 0 and len(validation['constraint_violations']) == 0
        
        return validation
        
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded dataset.
        
        Returns:
            Statistics dictionary
        """
        if not self.dataset:
            return {'empty': True}
            
        stats = {
            'total_samples': len(self.dataset),
            'field_coverage': {},
            'size_statistics': {},
            'protocol_types': {}
        }
        
        # Field coverage
        for field in self.protocol_fields:
            count = sum(1 for packet in self.dataset if field in packet)
            stats['field_coverage'][field] = count / len(self.dataset)
            
        # Size statistics
        sizes = []
        for packet in self.dataset:
            total_size = sum(
                len(v) if isinstance(v, bytes) else 1 
                for k, v in packet.items() 
                if not k.startswith('_')
            )
            sizes.append(total_size)
            
        if sizes:
            stats['size_statistics'] = {
                'mean': np.mean(sizes),
                'std': np.std(sizes),
                'min': np.min(sizes),
                'max': np.max(sizes)
            }
            
        return stats