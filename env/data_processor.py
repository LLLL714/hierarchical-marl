"""Protocol Data Processor for Fuzzing Framework.

This module handles preprocessing of protocol packet datasets and 
integration with Mamba feature extraction.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union


class ProtocolDataProcessor:
    """
    Processes protocol packet data and integrates Mamba feature extraction.
    
    Handles:
    - Protocol field extraction and structuring
    - Mamba feature integration
    - Data normalization and encoding
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.mamba_feature_dim = config.get('mamba_feature_dim', 256)
        self.max_field_length = config.get('max_field_length', 1024)
        self.supported_protocols = config.get('supported_protocols', ['tcp', 'udp', 'http', 'custom'])
        
        # Field type mappings
        self.field_type_mapping = {
            'header': 0,
            'payload': 1, 
            'checksum': 2,
            'length': 3,
            'address': 4,
            'flag': 5,
            'data': 6,
            'unknown': 7
        }
        
        # Initialize Mamba feature extractor (placeholder for actual implementation)
        self.mamba_extractor = self._initialize_mamba_extractor()
        
    def process_protocol_dataset(self, dataset_path: str) -> List[Dict]:
        """Process a protocol dataset from file.
        
        Args:
            dataset_path: Path to dataset file
            
        Returns:
            List of processed protocol packets
        """
        # In real implementation, this would load actual dataset
        # For now, generate synthetic data for testing
        return self._generate_synthetic_dataset()
    
    def extract_fields(self, protocol_data: Dict) -> List[Dict]:
        """Extract and structure protocol fields.
        
        Args:
            protocol_data: Raw protocol packet data
            
        Returns:
            List of structured field dictionaries
        """
        if 'fields' in protocol_data:
            # Data already has field structure
            fields = protocol_data['fields']
        else:
            # Parse raw protocol data
            fields = self._parse_protocol_fields(protocol_data)
            
        # Enhance fields with additional features
        enhanced_fields = []
        for i, field in enumerate(fields):
            enhanced_field = {
                'name': field.get('name', f'field_{i}'),
                'type': field.get('type', 'unknown'),
                'value': field.get('value', ''),
                'length': field.get('length', len(field.get('value', ''))),
                'position': i,
                'context_score': self._calculate_context_score(field, fields),
                'criticality': self._estimate_field_criticality(field),
                'mutability': self._estimate_field_mutability(field)
            }
            enhanced_fields.append(enhanced_field)
            
        return enhanced_fields
    
    def extract_mamba_features(self, protocol_data: Dict) -> np.ndarray:
        """Extract Mamba features from protocol data.
        
        Args:
            protocol_data: Protocol packet data
            
        Returns:
            Mamba feature vector
        """
        if self.mamba_extractor is None:
            # Return placeholder features if Mamba not available
            return self._generate_placeholder_mamba_features(protocol_data)
            
        # Convert protocol data to format expected by Mamba
        mamba_input = self._prepare_mamba_input(protocol_data)
        
        # Extract features using Mamba (placeholder implementation)
        features = self.mamba_extractor.extract_features(mamba_input)
        
        # Normalize and resize to expected dimension
        features = self._normalize_mamba_features(features)
        
        return features
    
    def encode_protocol_data(self, protocol_data: Dict) -> Dict:
        """Encode protocol data for neural network input.
        
        Args:
            protocol_data: Raw protocol data
            
        Returns:
            Encoded protocol data
        """
        encoded = {
            'protocol_type': self._encode_protocol_type(protocol_data.get('protocol_type', 'unknown')),
            'packet_size': min(len(protocol_data.get('raw_data', b'')), self.max_field_length),
            'field_count': len(protocol_data.get('fields', [])),
            'has_checksum': any(f.get('type') == 'checksum' for f in protocol_data.get('fields', [])),
            'has_payload': any(f.get('type') == 'payload' for f in protocol_data.get('fields', [])),
            'complexity_score': self._calculate_complexity_score(protocol_data)
        }
        
        return encoded
    
    def _initialize_mamba_extractor(self):
        """Initialize Mamba feature extractor.
        
        Returns:
            Mamba extractor instance or None if not available
        """
        # Placeholder for actual Mamba initialization
        # In real implementation, this would initialize the Mamba model
        try:
            # from mamba import MambaExtractor
            # return MambaExtractor(config=self.config)
            return MockMambaExtractor(self.mamba_feature_dim)
        except ImportError:
            print("Warning: Mamba extractor not available, using placeholder features")
            return None
    
    def _parse_protocol_fields(self, protocol_data: Dict) -> List[Dict]:
        """Parse protocol fields from raw data.
        
        Args:
            protocol_data: Raw protocol data
            
        Returns:
            List of parsed fields
        """
        raw_data = protocol_data.get('raw_data', b'')
        protocol_type = protocol_data.get('protocol_type', 'unknown')
        
        # Simple parsing logic (would be protocol-specific in real implementation)
        fields = []
        
        if protocol_type == 'tcp':
            fields = self._parse_tcp_fields(raw_data)
        elif protocol_type == 'udp':
            fields = self._parse_udp_fields(raw_data)
        elif protocol_type == 'http':
            fields = self._parse_http_fields(raw_data)
        else:
            fields = self._parse_generic_fields(raw_data)
            
        return fields
    
    def _parse_tcp_fields(self, raw_data: bytes) -> List[Dict]:
        """Parse TCP packet fields."""
        fields = []
        if len(raw_data) >= 20:  # Minimum TCP header size
            fields.extend([
                {'name': 'src_port', 'type': 'address', 'value': raw_data[0:2], 'length': 2},
                {'name': 'dst_port', 'type': 'address', 'value': raw_data[2:4], 'length': 2},
                {'name': 'seq_num', 'type': 'header', 'value': raw_data[4:8], 'length': 4},
                {'name': 'ack_num', 'type': 'header', 'value': raw_data[8:12], 'length': 4},
                {'name': 'flags', 'type': 'flag', 'value': raw_data[12:14], 'length': 2},
                {'name': 'window', 'type': 'header', 'value': raw_data[14:16], 'length': 2},
                {'name': 'checksum', 'type': 'checksum', 'value': raw_data[16:18], 'length': 2},
                {'name': 'urgent', 'type': 'header', 'value': raw_data[18:20], 'length': 2}
            ])
            if len(raw_data) > 20:
                fields.append({'name': 'payload', 'type': 'payload', 'value': raw_data[20:], 'length': len(raw_data) - 20})
        return fields
    
    def _parse_udp_fields(self, raw_data: bytes) -> List[Dict]:
        """Parse UDP packet fields."""
        fields = []
        if len(raw_data) >= 8:  # UDP header size
            fields.extend([
                {'name': 'src_port', 'type': 'address', 'value': raw_data[0:2], 'length': 2},
                {'name': 'dst_port', 'type': 'address', 'value': raw_data[2:4], 'length': 2},
                {'name': 'length', 'type': 'length', 'value': raw_data[4:6], 'length': 2},
                {'name': 'checksum', 'type': 'checksum', 'value': raw_data[6:8], 'length': 2}
            ])
            if len(raw_data) > 8:
                fields.append({'name': 'payload', 'type': 'payload', 'value': raw_data[8:], 'length': len(raw_data) - 8})
        return fields
    
    def _parse_http_fields(self, raw_data: bytes) -> List[Dict]:
        """Parse HTTP packet fields."""
        try:
            data_str = raw_data.decode('utf-8', errors='ignore')
            lines = data_str.split('\n')
            fields = []
            
            if lines:
                # HTTP request/response line
                fields.append({'name': 'request_line', 'type': 'header', 'value': lines[0].encode(), 'length': len(lines[0])})
                
                # Headers
                header_end = 0
                for i, line in enumerate(lines[1:], 1):
                    if line.strip() == '':
                        header_end = i
                        break
                    fields.append({'name': f'header_{i}', 'type': 'header', 'value': line.encode(), 'length': len(line)})
                
                # Body
                if header_end > 0 and header_end < len(lines) - 1:
                    body = '\n'.join(lines[header_end+1:])
                    fields.append({'name': 'body', 'type': 'payload', 'value': body.encode(), 'length': len(body)})
                    
            return fields
        except Exception:
            return self._parse_generic_fields(raw_data)
    
    def _parse_generic_fields(self, raw_data: bytes) -> List[Dict]:
        """Parse generic packet fields."""
        fields = []
        chunk_size = min(8, len(raw_data) // 4) if len(raw_data) > 0 else 1
        
        for i in range(0, len(raw_data), chunk_size):
            chunk = raw_data[i:i+chunk_size]
            field_type = 'header' if i < len(raw_data) // 2 else 'payload'
            fields.append({
                'name': f'field_{i//chunk_size}',
                'type': field_type,
                'value': chunk,
                'length': len(chunk)
            })
            
        return fields
    
    def _calculate_context_score(self, field: Dict, all_fields: List[Dict]) -> float:
        """Calculate context score for a field based on its relationship to other fields.
        
        Args:
            field: Current field
            all_fields: All fields in the packet
            
        Returns:
            Context score between 0 and 1
        """
        score = 0.5  # Base score
        
        # Position-based scoring
        position = field.get('position', 0)
        total_fields = len(all_fields)
        if total_fields > 1:
            position_score = 1.0 - (position / (total_fields - 1))  # Earlier fields more important
            score += 0.2 * position_score
        
        # Type-based scoring
        field_type = field.get('type', 'unknown')
        if field_type in ['header', 'checksum']:
            score += 0.2  # More critical fields
        elif field_type == 'payload':
            score += 0.1  # Moderately important
            
        # Length-based scoring
        field_length = field.get('length', 0)
        if field_length > 0:
            # Longer fields might be more complex and interesting
            length_score = min(field_length / 100.0, 0.2)
            score += length_score
            
        return min(score, 1.0)
    
    def _estimate_field_criticality(self, field: Dict) -> float:
        """Estimate how critical a field is to protocol operation.
        
        Args:
            field: Field dictionary
            
        Returns:
            Criticality score between 0 and 1
        """
        field_type = field.get('type', 'unknown')
        field_name = field.get('name', '').lower()
        
        # Type-based criticality
        if field_type == 'checksum':
            return 0.9
        elif field_type in ['header', 'length']:
            return 0.8
        elif field_type == 'address':
            return 0.7
        elif field_type == 'flag':
            return 0.6
        elif field_type == 'payload':
            return 0.4
        else:
            return 0.3
    
    def _estimate_field_mutability(self, field: Dict) -> float:
        """Estimate how suitable a field is for mutation.
        
        Args:
            field: Field dictionary
            
        Returns:
            Mutability score between 0 and 1
        """
        field_type = field.get('type', 'unknown')
        field_length = field.get('length', 0)
        
        # Longer fields generally more mutable
        length_score = min(field_length / 50.0, 0.5)
        
        # Type-based mutability
        if field_type == 'payload':
            type_score = 0.8
        elif field_type in ['data', 'unknown']:
            type_score = 0.7
        elif field_type in ['header', 'flag']:
            type_score = 0.6
        elif field_type == 'address':
            type_score = 0.5
        elif field_type in ['checksum', 'length']:
            type_score = 0.3  # Mutating these might break protocol
        else:
            type_score = 0.4
            
        return min(length_score + type_score, 1.0)
    
    def _encode_protocol_type(self, protocol_type: str) -> int:
        """Encode protocol type as integer.
        
        Args:
            protocol_type: Protocol type string
            
        Returns:
            Encoded protocol type
        """
        protocol_mapping = {
            'tcp': 0,
            'udp': 1,
            'http': 2,
            'https': 3,
            'ftp': 4,
            'smtp': 5,
            'dns': 6,
            'custom': 7,
            'unknown': 8
        }
        return protocol_mapping.get(protocol_type.lower(), 8)
    
    def _calculate_complexity_score(self, protocol_data: Dict) -> float:
        """Calculate complexity score for protocol data.
        
        Args:
            protocol_data: Protocol data dictionary
            
        Returns:
            Complexity score between 0 and 1
        """
        fields = protocol_data.get('fields', [])
        raw_data = protocol_data.get('raw_data', b'')
        
        # Number of fields
        field_complexity = min(len(fields) / 20.0, 0.4)
        
        # Data size complexity
        size_complexity = min(len(raw_data) / 1500.0, 0.3)  # 1500 bytes as typical MTU
        
        # Field type diversity
        unique_types = len(set(f.get('type', 'unknown') for f in fields))
        type_complexity = min(unique_types / 8.0, 0.3)  # 8 is max field types
        
        return field_complexity + size_complexity + type_complexity
    
    def _generate_placeholder_mamba_features(self, protocol_data: Dict) -> np.ndarray:
        """Generate placeholder Mamba features when Mamba is not available.
        
        Args:
            protocol_data: Protocol data
            
        Returns:
            Placeholder feature vector
        """
        # Generate features based on protocol characteristics
        features = []
        
        # Protocol type encoding
        protocol_type = self._encode_protocol_type(protocol_data.get('protocol_type', 'unknown'))
        features.extend([protocol_type / 8.0])  # Normalize
        
        # Size-based features
        raw_data = protocol_data.get('raw_data', b'')
        size_features = [
            len(raw_data) / 1500.0,  # Normalized packet size
            min(len(raw_data), 100) / 100.0,  # Capped size feature
        ]
        features.extend(size_features)
        
        # Field-based features
        fields = protocol_data.get('fields', [])
        field_features = [
            len(fields) / 20.0,  # Number of fields
            sum(1 for f in fields if f.get('type') == 'header') / max(len(fields), 1),
            sum(1 for f in fields if f.get('type') == 'payload') / max(len(fields), 1),
            sum(f.get('length', 0) for f in fields) / max(len(raw_data), 1)
        ]
        features.extend(field_features)
        
        # Statistical features
        if raw_data:
            byte_values = list(raw_data)
            stat_features = [
                np.mean(byte_values) / 255.0,
                np.std(byte_values) / 255.0,
                len(set(byte_values)) / 256.0,  # Unique byte ratio
            ]
            features.extend(stat_features)
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Pad or truncate to desired dimension
        if len(features) < self.mamba_feature_dim:
            # Add random but deterministic features based on data hash
            np.random.seed(hash(str(protocol_data)) % 2**32)
            padding = np.random.normal(0, 0.1, self.mamba_feature_dim - len(features))
            features.extend(padding)
        
        features = np.array(features[:self.mamba_feature_dim], dtype=np.float32)
        return features
    
    def _prepare_mamba_input(self, protocol_data: Dict) -> Any:
        """Prepare input for Mamba feature extraction.
        
        Args:
            protocol_data: Protocol data
            
        Returns:
            Formatted input for Mamba
        """
        # Convert protocol data to sequence format expected by Mamba
        raw_data = protocol_data.get('raw_data', b'')
        
        # Convert bytes to sequence of integers
        sequence = list(raw_data)
        
        # Limit sequence length
        max_sequence_length = 512
        if len(sequence) > max_sequence_length:
            sequence = sequence[:max_sequence_length]
        elif len(sequence) < max_sequence_length:
            sequence.extend([0] * (max_sequence_length - len(sequence)))
            
        return np.array(sequence, dtype=np.int32)
    
    def _normalize_mamba_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize Mamba features to expected format.
        
        Args:
            features: Raw Mamba features
            
        Returns:
            Normalized features
        """
        # Resize to expected dimension
        if len(features) > self.mamba_feature_dim:
            features = features[:self.mamba_feature_dim]
        elif len(features) < self.mamba_feature_dim:
            features = np.pad(features, (0, self.mamba_feature_dim - len(features)))
        
        # Normalize to [-1, 1] range
        if np.std(features) > 0:
            features = (features - np.mean(features)) / np.std(features)
            features = np.clip(features, -3, 3) / 3.0  # Clip outliers and normalize
        
        return features.astype(np.float32)
    
    def _generate_synthetic_dataset(self) -> List[Dict]:
        """Generate synthetic protocol dataset for testing.
        
        Returns:
            List of synthetic protocol packets
        """
        dataset = []
        
        # Generate different protocol types
        for protocol_type in ['tcp', 'udp', 'http']:
            for i in range(10):  # 10 packets per protocol
                if protocol_type == 'tcp':
                    raw_data = self._generate_synthetic_tcp_packet()
                elif protocol_type == 'udp':
                    raw_data = self._generate_synthetic_udp_packet()
                else:  # http
                    raw_data = self._generate_synthetic_http_packet()
                
                packet = {
                    'protocol_type': protocol_type,
                    'raw_data': raw_data,
                    'packet_id': f'{protocol_type}_{i}'
                }
                dataset.append(packet)
        
        return dataset
    
    def _generate_synthetic_tcp_packet(self) -> bytes:
        """Generate synthetic TCP packet."""
        # TCP header (20 bytes) + payload
        header = bytes([
            0x00, 0x50,  # src port (80)
            0x1F, 0x90,  # dst port (8080)
            0x00, 0x00, 0x00, 0x01,  # seq num
            0x00, 0x00, 0x00, 0x00,  # ack num
            0x50, 0x02,  # header length + flags
            0xFF, 0xFF,  # window size
            0x00, 0x00,  # checksum
            0x00, 0x00   # urgent pointer
        ])
        payload = b'Hello, World!'
        return header + payload
    
    def _generate_synthetic_udp_packet(self) -> bytes:
        """Generate synthetic UDP packet."""
        # UDP header (8 bytes) + payload
        payload = b'UDP test data'
        header = bytes([
            0x00, 0x35,  # src port (53)
            0x00, 0x35,  # dst port (53)
            0x00, len(payload) + 8,  # length
            0x00, 0x00   # checksum
        ])
        return header + payload
    
    def _generate_synthetic_http_packet(self) -> bytes:
        """Generate synthetic HTTP packet."""
        http_request = (
            "GET /index.html HTTP/1.1\r\n"
            "Host: example.com\r\n"
            "User-Agent: TestAgent/1.0\r\n"
            "Accept: text/html\r\n"
            "\r\n"
        )
        return http_request.encode('utf-8')


class MockMambaExtractor:
    """Mock Mamba extractor for testing purposes."""
    
    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
    
    def extract_features(self, input_data: np.ndarray) -> np.ndarray:
        """Extract mock features from input data."""
        # Generate deterministic but varied features based on input
        np.random.seed(hash(input_data.tobytes()) % 2**32)
        features = np.random.normal(0, 1, self.feature_dim).astype(np.float32)
        return features