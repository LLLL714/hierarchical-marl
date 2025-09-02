"""Protocol Data Processor for Fuzzing Framework.

This module handles loading, preprocessing, and feature extraction for protocol datasets
used in the hierarchical fuzzing reinforcement learning framework.
"""

import numpy as np
import json
import os
import pickle
import random
from typing import Dict, List, Any, Optional, Tuple
import hashlib


class ProtocolDataProcessor:
    """Processes protocol data for fuzzing with Mamba feature integration."""
    
    def __init__(self, config: Dict):
        """Initialize data processor.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config
        self.mamba_feature_dim = config.get('mamba_feature_dim', 256)
        self.protocol_fields = config.get('protocol_fields', [])
        
        # Cache for feature extraction
        self._feature_cache = {}
        self._packet_cache = {}
        
        # Default protocol field definitions if not provided
        if not self.protocol_fields:
            self.protocol_fields = self._get_default_protocol_fields()
            
    def _get_default_protocol_fields(self) -> List[Dict]:
        """Get default protocol field definitions for common protocols.
        
        Returns:
            List of protocol field dictionaries
        """
        return [
            {'name': 'header', 'type': 'bytes', 'min_length': 1, 'max_length': 64},
            {'name': 'payload', 'type': 'bytes', 'min_length': 0, 'max_length': 1500},
            {'name': 'length_field', 'type': 'integer', 'min_value': 0, 'max_value': 65535},
            {'name': 'type_field', 'type': 'integer', 'min_value': 0, 'max_value': 255},
            {'name': 'flags', 'type': 'bitfield', 'num_bits': 8},
            {'name': 'checksum', 'type': 'integer', 'min_value': 0, 'max_value': 4294967295},
            {'name': 'sequence_number', 'type': 'integer', 'min_value': 0, 'max_value': 4294967295},
            {'name': 'options', 'type': 'bytes', 'min_length': 0, 'max_length': 40}
        ]
        
    def load_protocol_dataset(self, dataset_path: str) -> List[Dict]:
        """Load preprocessed protocol dataset from file.
        
        Args:
            dataset_path: Path to dataset file
            
        Returns:
            List of protocol packet dictionaries
        """
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset path {dataset_path} not found. Generating synthetic data.")
            return self.generate_synthetic_dataset(100)
            
        try:
            with open(dataset_path, 'rb') as f:
                if dataset_path.endswith('.json'):
                    return json.load(f)
                elif dataset_path.endswith('.pkl'):
                    return pickle.load(f)
                else:
                    # Assume binary format, parse accordingly
                    return self._parse_binary_dataset(f.read())
        except Exception as e:
            print(f"Error loading dataset: {e}. Generating synthetic data.")
            return self.generate_synthetic_dataset(100)
            
    def load_random_packet(self, dataset_path: str) -> Dict:
        """Load a random packet from the dataset.
        
        Args:
            dataset_path: Path to dataset file
            
        Returns:
            Random protocol packet dictionary
        """
        # Check cache first
        cache_key = f"dataset_{dataset_path}"
        if cache_key not in self._packet_cache:
            self._packet_cache[cache_key] = self.load_protocol_dataset(dataset_path)
            
        dataset = self._packet_cache[cache_key]
        if not dataset:
            return self.generate_default_packet()
            
        return random.choice(dataset)
        
    def generate_default_packet(self) -> Dict:
        """Generate a default protocol packet for initialization.
        
        Returns:
            Default packet dictionary
        """
        packet = {}
        
        for field in self.protocol_fields:
            field_name = field['name']
            field_type = field.get('type', 'bytes')
            
            if field_type == 'bytes':
                min_len = field.get('min_length', 1)
                max_len = field.get('max_length', 64)
                length = random.randint(min_len, max_len)
                packet[field_name] = np.random.randint(0, 256, size=length, dtype=np.uint8).tobytes()
                
            elif field_type == 'integer':
                min_val = field.get('min_value', 0)
                max_val = field.get('max_value', 65535)
                packet[field_name] = random.randint(min_val, max_val)
                
            elif field_type == 'bitfield':
                num_bits = field.get('num_bits', 8)
                packet[field_name] = random.randint(0, (1 << num_bits) - 1)
                
        return packet
        
    def generate_synthetic_dataset(self, num_packets: int) -> List[Dict]:
        """Generate synthetic protocol dataset for testing.
        
        Args:
            num_packets: Number of packets to generate
            
        Returns:
            List of synthetic protocol packets
        """
        dataset = []
        
        for _ in range(num_packets):
            packet = self.generate_default_packet()
            
            # Add some variation patterns
            if random.random() < 0.3:  # 30% chance of larger packets
                for field in self.protocol_fields:
                    if field.get('type') == 'bytes' and field['name'] in packet:
                        current_data = packet[field['name']]
                        extension = np.random.randint(0, 256, size=random.randint(10, 100), dtype=np.uint8).tobytes()
                        packet[field['name']] = current_data + extension
                        
            if random.random() < 0.2:  # 20% chance of minimal packets
                for field in self.protocol_fields:
                    if field.get('type') == 'bytes' and field['name'] in packet:
                        min_len = field.get('min_length', 1)
                        packet[field['name']] = np.random.randint(0, 256, size=min_len, dtype=np.uint8).tobytes()
                        
            dataset.append(packet)
            
        return dataset
        
    def extract_mamba_features(self, packet: Dict) -> np.ndarray:
        """Extract Mamba features from protocol packet.
        
        Args:
            packet: Protocol packet dictionary
            
        Returns:
            Mamba feature vector
        """
        # Generate cache key for packet
        packet_str = str(sorted(packet.items()))
        cache_key = hashlib.md5(packet_str.encode()).hexdigest()
        
        # Check cache
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
            
        # Simulate Mamba feature extraction
        # In real implementation, this would interface with actual Mamba model
        features = self._simulate_mamba_features(packet)
        
        # Cache results
        self._feature_cache[cache_key] = features
        
        return features
        
    def _simulate_mamba_features(self, packet: Dict) -> np.ndarray:
        """Simulate Mamba feature extraction for protocol packet.
        
        Args:
            packet: Protocol packet dictionary
            
        Returns:
            Simulated feature vector
        """
        # Create deterministic features based on packet content
        features = np.zeros(self.mamba_feature_dim)
        
        # Extract structural features
        feature_idx = 0
        
        for field in self.protocol_fields:
            field_name = field['name']
            if field_name in packet and feature_idx < self.mamba_feature_dim:
                value = packet[field_name]
                
                if isinstance(value, bytes):
                    # For byte fields, use statistical features
                    if len(value) > 0:
                        # Basic statistics
                        features[feature_idx] = len(value) / 1500.0  # Normalized length
                        feature_idx += 1
                        
                        if feature_idx < self.mamba_feature_dim:
                            features[feature_idx] = np.mean(list(value)) / 255.0  # Mean byte value
                            feature_idx += 1
                            
                        if feature_idx < self.mamba_feature_dim:
                            features[feature_idx] = np.std(list(value)) / 255.0  # Std of byte values
                            feature_idx += 1
                            
                        # Entropy-like measure
                        if feature_idx < self.mamba_feature_dim:
                            byte_counts = np.bincount(list(value), minlength=256)
                            probabilities = byte_counts / len(value)
                            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                            features[feature_idx] = entropy / 8.0  # Normalized entropy
                            feature_idx += 1
                            
                elif isinstance(value, int):
                    # For integer fields, normalize and store
                    if feature_idx < self.mamba_feature_dim:
                        max_val = field.get('max_value', 65535)
                        features[feature_idx] = value / max_val
                        feature_idx += 1
                        
        # Fill remaining features with packet-level statistics
        while feature_idx < self.mamba_feature_dim:
            # Use packet hash to generate deterministic but varied features
            packet_hash = hash(packet_str) % (2**32)
            seed_val = (packet_hash + feature_idx) % 1000
            features[feature_idx] = seed_val / 1000.0
            feature_idx += 1
            
        # Add some temporal and contextual features
        if feature_idx < self.mamba_feature_dim:
            # Packet complexity measure
            total_bytes = sum(len(v) if isinstance(v, bytes) else 4 for v in packet.values())
            features[feature_idx-1] = min(1.0, total_bytes / 2000.0)
            
        return features
        
    def _parse_binary_dataset(self, data: bytes) -> List[Dict]:
        """Parse binary dataset format.
        
        Args:
            data: Raw binary data
            
        Returns:
            List of parsed packets
        """
        # Simplified binary parsing - in real implementation, this would
        # parse actual protocol formats (pcap, etc.)
        packets = []
        
        # Split data into chunks and create packets
        chunk_size = 64
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            if len(chunk) < 10:  # Skip small chunks
                continue
                
            packet = {
                'header': chunk[:8],
                'payload': chunk[8:],
                'length_field': len(chunk),
                'type_field': chunk[0] if chunk else 0
            }
            packets.append(packet)
            
        return packets[:100]  # Limit for performance
        
    def extract_field_features(self, packet: Dict, field_name: str) -> Dict:
        """Extract detailed features for a specific protocol field.
        
        Args:
            packet: Protocol packet dictionary
            field_name: Name of field to analyze
            
        Returns:
            Field feature dictionary
        """
        if field_name not in packet:
            return {'exists': False}
            
        value = packet[field_name]
        features = {'exists': True}
        
        if isinstance(value, bytes):
            features.update({
                'length': len(value),
                'mean_byte_value': np.mean(list(value)) if value else 0,
                'std_byte_value': np.std(list(value)) if value else 0,
                'min_byte_value': min(value) if value else 0,
                'max_byte_value': max(value) if value else 0,
                'entropy': self._calculate_entropy(value),
                'has_null_bytes': b'\x00' in value,
                'has_printable': any(32 <= b <= 126 for b in value)
            })
        elif isinstance(value, int):
            features.update({
                'value': value,
                'is_power_of_2': value > 0 and (value & (value - 1)) == 0,
                'bit_count': bin(value).count('1'),
                'is_common_value': value in [0, 1, 255, 65535, 4294967295]
            })
            
        return features
        
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of byte data.
        
        Args:
            data: Byte data
            
        Returns:
            Entropy value
        """
        if not data:
            return 0.0
            
        byte_counts = np.bincount(list(data), minlength=256)
        probabilities = byte_counts / len(data)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return entropy
        
    def preprocess_packet_for_mutation(self, packet: Dict) -> Dict:
        """Preprocess packet to prepare for mutation operations.
        
        Args:
            packet: Original packet dictionary
            
        Returns:
            Preprocessed packet ready for mutation
        """
        processed_packet = {}
        
        for field_name, value in packet.items():
            if isinstance(value, int):
                # Convert integers to bytes for uniform mutation handling
                if value <= 255:
                    processed_packet[field_name] = value.to_bytes(1, 'big')
                elif value <= 65535:
                    processed_packet[field_name] = value.to_bytes(2, 'big')
                else:
                    processed_packet[field_name] = value.to_bytes(4, 'big')
            else:
                processed_packet[field_name] = value
                
        return processed_packet
        
    def postprocess_mutated_packet(self, packet: Dict, original_packet: Dict) -> Dict:
        """Postprocess mutated packet to restore proper types.
        
        Args:
            packet: Mutated packet dictionary
            original_packet: Original packet before mutation
            
        Returns:
            Postprocessed packet with correct types
        """
        processed_packet = {}
        
        for field_name, value in packet.items():
            if field_name in original_packet:
                original_value = original_packet[field_name]
                
                if isinstance(original_value, int) and isinstance(value, bytes):
                    # Convert bytes back to integer
                    try:
                        processed_packet[field_name] = int.from_bytes(value[:4], 'big')
                    except ValueError:
                        processed_packet[field_name] = 0
                else:
                    processed_packet[field_name] = value
            else:
                processed_packet[field_name] = value
                
        return processed_packet
        
    def get_field_context(self, packet: Dict, field_name: str) -> Dict:
        """Get contextual information about a field within the packet.
        
        Args:
            packet: Protocol packet dictionary
            field_name: Name of field to analyze
            
        Returns:
            Context dictionary
        """
        context = {
            'field_position': list(packet.keys()).index(field_name) if field_name in packet else -1,
            'total_fields': len(packet),
            'packet_size': sum(len(v) if isinstance(v, bytes) else 4 for v in packet.values()),
            'field_relationship': self._analyze_field_relationships(packet, field_name)
        }
        
        return context
        
    def _analyze_field_relationships(self, packet: Dict, field_name: str) -> Dict:
        """Analyze relationships between fields in the packet.
        
        Args:
            packet: Protocol packet dictionary
            field_name: Target field name
            
        Returns:
            Relationship analysis dictionary
        """
        if field_name not in packet:
            return {}
            
        target_value = packet[field_name]
        relationships = {
            'depends_on_length': False,
            'affects_checksum': False,
            'sequence_dependency': False
        }
        
        # Simple heuristic analysis
        if 'length' in field_name.lower():
            relationships['depends_on_length'] = True
            
        if 'checksum' in field_name.lower() or 'crc' in field_name.lower():
            relationships['affects_checksum'] = True
            
        if 'seq' in field_name.lower() or 'sequence' in field_name.lower():
            relationships['sequence_dependency'] = True
            
        return relationships
        
    def save_processed_dataset(self, dataset: List[Dict], output_path: str):
        """Save processed dataset to file.
        
        Args:
            dataset: List of processed packets
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            if output_path.endswith('.json'):
                with open(output_path, 'w') as f:
                    json.dump(dataset, f, indent=2, default=str)
            else:
                with open(output_path, 'wb') as f:
                    pickle.dump(dataset, f)
        except Exception as e:
            print(f"Error saving dataset: {e}")
            
    def get_dataset_statistics(self, dataset: List[Dict]) -> Dict:
        """Get statistical analysis of the dataset.
        
        Args:
            dataset: List of protocol packets
            
        Returns:
            Statistics dictionary
        """
        if not dataset:
            return {}
            
        stats = {
            'total_packets': len(dataset),
            'field_statistics': {},
            'size_statistics': {
                'min_packet_size': float('inf'),
                'max_packet_size': 0,
                'avg_packet_size': 0
            }
        }
        
        total_size = 0
        for packet in dataset:
            packet_size = sum(len(v) if isinstance(v, bytes) else 4 for v in packet.values())
            total_size += packet_size
            stats['size_statistics']['min_packet_size'] = min(stats['size_statistics']['min_packet_size'], packet_size)
            stats['size_statistics']['max_packet_size'] = max(stats['size_statistics']['max_packet_size'], packet_size)
            
        stats['size_statistics']['avg_packet_size'] = total_size / len(dataset)
        
        # Field occurrence statistics
        for packet in dataset:
            for field_name in packet.keys():
                if field_name not in stats['field_statistics']:
                    stats['field_statistics'][field_name] = {
                        'occurrence_count': 0,
                        'avg_size': 0,
                        'size_variance': 0
                    }
                stats['field_statistics'][field_name]['occurrence_count'] += 1
                
        return stats