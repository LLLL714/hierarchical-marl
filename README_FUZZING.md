# Hierarchical Fuzzing Reinforcement Learning Framework

This repository contains the implementation of a hierarchical reinforcement learning framework adapted for intelligent protocol fuzzing. The framework is based on the Hierarchical Cooperative Multi-Agent Reinforcement Learning with Skill Discovery (HSD) algorithm, transformed for the fuzzing domain.

## Overview

The original HSD framework for multi-agent sports simulation has been completely transformed into a single-agent hierarchical fuzzing system:

- **High-level Policy**: Protocol field selection (instead of agent role assignment)
- **Low-level Policy**: Mutation action selection (instead of action execution)  
- **Environment**: Protocol fuzzing (instead of sports simulation)
- **Decoder**: Field importance learning (instead of skill discovery)
- **Reward System**: 3-tier fuzzing effectiveness (instead of single reward)

## Architecture

### Hierarchical Decision Making

```
┌─────────────────────────────────────────────────────────┐
│                 High-Level Policy                       │
│          (Protocol Field Selection)                     │
│                                                         │
│ • 90% exploit best fields based on reward history      │
│ • 10% random exploration for diversity                 │
│ • Learns field importance via decoder network          │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                 Low-Level Policy                        │
│            (Mutation Action Selection)                  │
│                                                         │
│ • 6 mutation actions: insert, delete, flip,            │
│   replace, shuffle, copy                                │
│ • Context-aware based on field characteristics         │
│ • Considers field type, length, mutation history       │
└─────────────────────────────────────────────────────────┘
```

### Three-Tier Reward System

1. **Basic Rewards** (Weight: 1.0)
   - Code coverage increase
   - State novelty detection  
   - Unique packet retention
   - Execution stability

2. **Medium Rewards** (Weight: 5.0)
   - Crash detection (segfault, heap corruption)
   - Timeout/deadlock detection
   - Protocol state anomalies
   - Memory usage anomalies

3. **High Rewards** (Weight: 100.0)
   - Vulnerability discovery (buffer overflow, use-after-free, etc.)
   - Security boundary violations
   - Privilege escalation indicators

## Project Structure

```
├── alg/                          # Core algorithms
│   ├── alg_fuzzing_hsd.py       # Main fuzzing HSD algorithm
│   ├── networks_fuzzing.py      # Fuzzing-specific neural networks
│   ├── train_fuzzing.py         # Training script for fuzzing
│   ├── config_fuzzing.json      # Fuzzing configuration
│   └── [original HSD files]     # Original algorithm files
│
├── env/                          # Environment components
│   ├── fuzzing_environment.py   # Main fuzzing environment
│   ├── data_processor.py        # Protocol data processing & Mamba features
│   ├── reward_calculator.py     # 3-tier reward calculation
│   └── env_wrapper.py          # Original environment wrapper
│
├── test/                         # Testing framework
│   ├── test_fuzzing.py          # Comprehensive fuzzing tests
│   └── [original test files]    # Original tests
│
└── results/                      # Training results and logs
```

## Key Features

### 1. Mamba Feature Integration
- Extracts 256-dimensional features from protocol packets
- Provides semantic understanding of packet structure
- Enables intelligent field importance scoring

### 2. Intelligent Field Selection
- **90/10 Strategy**: 90% exploitation of high-reward fields, 10% random exploration
- Dynamic field importance scoring based on historical performance
- Curriculum learning: starts with fewer fields, expands based on decoder confidence

### 3. Context-Aware Mutation Actions
- **6 Mutation Types**: insert, delete, flip, replace, shuffle, copy
- Field-specific mutation strategies based on:
  - Field type (bytes, integer, bitfield)
  - Field length and entropy
  - Historical mutation success
  - Protocol field relationships

### 4. Advanced Reward Shaping
- Sparse reward problem solved through hierarchical reward structure
- Adaptive reward scaling based on discovery rates
- Detailed reward breakdown for analysis and debugging

## Installation and Setup

### Prerequisites
- Python >= 3.7
- TensorFlow 1.13+ (for compatibility with original HSD)
- NumPy
- Additional dependencies as needed

### Installation
```bash
git clone <repository-url>
cd hierarchical-marl
pip install -r requirements.txt  # If available
```

## Usage

### Training

```bash
cd alg
python train_fuzzing.py config_fuzzing.json
```

### Configuration

The `config_fuzzing.json` file contains all configuration parameters:

- **Protocol Fields**: Define the protocol structure to fuzz
- **Mutation Actions**: Configure available mutation strategies  
- **Neural Networks**: Set network architectures and hyperparameters
- **Reward Weights**: Adjust the 3-tier reward system
- **Training Parameters**: Control learning rates, exploration, etc.

### Testing

```bash
cd test
python test_fuzzing.py                    # Run all tests
python test_fuzzing.py --test environment # Run specific test suite
```

### Example Protocol Configuration

```json
{
  "protocol_fields": [
    {
      "name": "header",
      "type": "bytes",
      "min_length": 1,
      "max_length": 64,
      "importance_weight": 1.0
    },
    {
      "name": "payload", 
      "type": "bytes",
      "min_length": 0,
      "max_length": 1500,
      "importance_weight": 0.8
    }
  ]
}
```

## Performance Metrics

The framework tracks comprehensive metrics:

- **Coverage Metrics**: Code coverage increase, new states discovered
- **Discovery Metrics**: Crashes found, vulnerabilities detected
- **Efficiency Metrics**: Mutations per discovery, field exploration balance
- **Learning Metrics**: Field importance evolution, reward distribution

## Research Applications

This framework enables research in:

1. **Intelligent Fuzzing**: Learning-based fuzzing strategies
2. **Hierarchical RL**: Multi-level decision making in security
3. **Protocol Analysis**: Automated protocol field importance discovery
4. **Vulnerability Research**: Systematic vulnerability discovery methods

## Comparison with Original HSD

| Aspect | Original HSD | Fuzzing HSD |
|--------|-------------|-------------|
| Domain | Multi-agent sports | Single-agent fuzzing |
| High-level | Agent roles | Protocol fields |
| Low-level | Team actions | Mutation actions |
| Environment | STS2 simulator | Protocol fuzzing |
| Agents | 3-5 per team | Single fuzzing agent |
| Rewards | Game scoring | Coverage/crashes/vulnerabilities |
| Skills | Coordination patterns | Field importance patterns |

## Evaluation

The framework includes comprehensive evaluation metrics:

- **Fuzzing Effectiveness**: Crash and vulnerability discovery rates
- **Coverage Efficiency**: Coverage growth per mutation
- **Learning Progress**: Field importance learning convergence
- **Exploration Balance**: Field exploration vs exploitation ratio

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests for new features
4. Ensure all tests pass
5. Submit a pull request

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{fuzzing_hsd_2024,
  title={Hierarchical Reinforcement Learning for Intelligent Protocol Fuzzing},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project is distributed under the same license as the original HSD implementation (BSD-3).

## Original HSD Citation

The original HSD algorithm this work is based on:

```bibtex
@inproceedings{yang2020hierarchical,
  title={Hierarchical Cooperative Multi-Agent Reinforcement Learning with Skill Discovery},
  author={Yang, Jiachen and Borovikov, Igor and Zha, Hongyuan},
  booktitle={Proceedings of the 19th International Conference on Autonomous Agents and MultiAgent Systems},
  pages={1566--1574},
  year={2020}
}
```

## Contact

For questions and support, please open an issue in the repository.