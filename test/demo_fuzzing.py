"""Simple demonstration of the Fuzzing HSD Framework.

This demonstrates the core fuzzing framework without TensorFlow dependencies.
"""

import sys
import os
import numpy as np

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'env'))

import fuzzing_environment
import data_processor
import reward_calculator


def demonstrate_fuzzing_framework():
    """Demonstrate the fuzzing framework components."""
    
    print("="*60)
    print("FUZZING HSD FRAMEWORK DEMONSTRATION")
    print("="*60)
    
    # Configuration
    config = {
        'protocol_fields': [
            {'name': 'header', 'type': 'header'},
            {'name': 'source_addr', 'type': 'address'},
            {'name': 'dest_addr', 'type': 'address'},
            {'name': 'length', 'type': 'length'},
            {'name': 'checksum', 'type': 'checksum'},
            {'name': 'payload', 'type': 'payload'}
        ],
        'mutation_actions': ['insert', 'delete', 'flip', 'replace', 'shuffle', 'copy'],
        'mamba_feature_dim': 256,
        'use_mamba_features': True,
        'exploit_ratio': 0.9,
        'max_steps': 50,
        'base_reward_weight': 1.0,
        'medium_reward_weight': 5.0,
        'high_reward_weight': 100.0,
        'coverage_reward_scale': 10.0,
        'crash_reward': 20.0,
        'vulnerability_reward': 1000.0
    }
    
    print("\n1. INITIALIZING COMPONENTS")
    print("-" * 30)
    
    # Initialize environment
    env = fuzzing_environment.FuzzingEnvironment(config)
    print(f"✓ Fuzzing Environment initialized")
    print(f"  - Protocol fields: {env.n_fields}")
    print(f"  - Mutation actions: {env.n_mutations}")
    print(f"  - State dimension: {env.state_dim}")
    print(f"  - Exploit ratio: {env.exploit_ratio}")
    
    # Initialize data processor
    processor = data_processor.ProtocolDataProcessor(config)
    print(f"✓ Data Processor initialized")
    print(f"  - Mamba feature dim: {processor.mamba_feature_dim}")
    print(f"  - Supported protocols: {processor.supported_protocols}")
    
    # Initialize reward calculator
    calculator = reward_calculator.RewardCalculator(config)
    print(f"✓ Reward Calculator initialized")
    print(f"  - Base weight: {calculator.base_reward_weight}")
    print(f"  - Medium weight: {calculator.medium_reward_weight}")
    print(f"  - High weight: {calculator.high_reward_weight}")
    
    print("\n2. DEMONSTRATING FIELD SELECTION (90/10 STRATEGY)")
    print("-" * 50)
    
    # Reset environment
    state, info = env.reset()
    print(f"✓ Environment reset")
    print(f"  - Initial state shape: {state.shape}")
    print(f"  - Protocol type: {info.get('protocol_type', 'unknown')}")
    
    # Demonstrate field selection probabilities
    field_probs = env.get_field_selection_probabilities()
    print(f"\n✓ Field selection probabilities:")
    for i, prob in enumerate(field_probs):
        field_name = config['protocol_fields'][i]['name']
        print(f"  - {field_name}: {prob:.3f}")
    
    # Show exploit vs explore breakdown
    exploit_component = 0.9 * (env.field_importance_history / np.sum(env.field_importance_history))
    explore_component = 0.1 * np.ones(env.n_fields) / env.n_fields
    print(f"\n✓ 90/10 Strategy breakdown:")
    print(f"  - Exploit (90%): {exploit_component}")
    print(f"  - Explore (10%): {explore_component}")
    
    print("\n3. DEMONSTRATING MUTATION ACTIONS")
    print("-" * 35)
    
    # Show available mutation actions
    print(f"✓ Available mutation actions: {env.mutation_actions}")
    
    # Demonstrate mutation action feature extraction
    for field_idx in range(min(3, env.n_fields)):
        features = env.get_mutation_action_features(field_idx)
        field_name = config['protocol_fields'][field_idx]['name']
        print(f"  - {field_name} features shape: {features.shape}")
    
    print("\n4. RUNNING SIMULATION EPISODE")
    print("-" * 32)
    
    total_reward = 0
    total_coverage = 0
    crashes_found = 0
    vulnerabilities_found = 0
    
    for step in range(20):
        # Select field using probabilities (90/10 strategy)
        field_probs = env.get_field_selection_probabilities()
        field_idx = np.random.choice(env.n_fields, p=field_probs)
        
        # Select mutation action based on field features
        field_features = env.get_mutation_action_features(field_idx)
        mutation_history = env._get_mutation_history_for_field_type('unknown')
        
        # Simple heuristic: choose mutation with highest historical reward + noise
        mutation_action = np.argmax(mutation_history + np.random.normal(0, 0.1, env.n_mutations))
        mutation_action = max(0, min(mutation_action, env.n_mutations - 1))
        
        # Execute step
        next_state, reward, done, step_info = env.step(field_idx, mutation_action)
        
        # Accumulate metrics
        total_reward += reward
        total_coverage += step_info.get('coverage_increase', 0)
        if step_info.get('crash_detected', False):
            crashes_found += 1
        if step_info.get('vulnerability_found', False):
            vulnerabilities_found += 1
        
        # Print step info
        field_name = config['protocol_fields'][field_idx]['name']
        mutation_name = env.mutation_actions[mutation_action]
        print(f"  Step {step+1:2d}: {field_name:12s} + {mutation_name:8s} → reward={reward:6.2f}")
        
        if done:
            print(f"    Episode terminated early!")
            break
        
        state = next_state
    
    print(f"\n✓ Episode completed!")
    print(f"  - Total steps: {step + 1}")
    print(f"  - Total reward: {total_reward:.2f}")
    print(f"  - Coverage increase: {total_coverage:.3f}")
    print(f"  - Crashes found: {crashes_found}")
    print(f"  - Vulnerabilities found: {vulnerabilities_found}")
    
    print("\n5. DEMONSTRATING THREE-TIER REWARD SYSTEM")
    print("-" * 42)
    
    # Test different mutation results
    test_cases = [
        {
            'name': 'Basic Coverage',
            'result': {'success': True, 'coverage_increase': 0.15, 'crash_detected': False, 'vulnerability_found': False}
        },
        {
            'name': 'Crash Detection',
            'result': {'success': True, 'coverage_increase': 0.05, 'crash_detected': True, 'vulnerability_found': False}
        },
        {
            'name': 'Vulnerability Discovery',
            'result': {'success': True, 'coverage_increase': 0.1, 'crash_detected': False, 'vulnerability_found': True}
        }
    ]
    
    for case in test_cases:
        reward = calculator.calculate_reward(case['result'])
        print(f"  - {case['name']:20s}: {reward:8.2f}")
    
    print("\n6. FIELD IMPORTANCE LEARNING")
    print("-" * 28)
    
    print("✓ Field importance evolution:")
    initial_importance = env.field_importance_history.copy()
    print(f"  - Initial: {initial_importance}")
    
    # Simulate learning by executing more steps
    for _ in range(10):
        field_idx = np.random.randint(0, env.n_fields)
        mutation_action = np.random.randint(0, env.n_mutations)
        env.step(field_idx, mutation_action)
    
    final_importance = env.field_importance_history.copy()
    print(f"  - Final:   {final_importance}")
    print(f"  - Change:  {final_importance - initial_importance}")
    
    print("\n7. DATA PROCESSING DEMONSTRATION")
    print("-" * 33)
    
    # Generate synthetic protocol data
    synthetic_data = processor._generate_synthetic_dataset()
    print(f"✓ Generated {len(synthetic_data)} synthetic protocol packets")
    
    # Process a sample packet
    sample_packet = synthetic_data[0]
    print(f"  - Sample protocol: {sample_packet['protocol_type']}")
    print(f"  - Raw data length: {len(sample_packet['raw_data'])} bytes")
    
    # Extract fields
    fields = processor.extract_fields(sample_packet)
    print(f"  - Extracted {len(fields)} fields:")
    for field in fields[:3]:  # Show first 3 fields
        print(f"    * {field['name']} ({field['type']}): criticality={field['criticality']:.2f}")
    
    # Extract Mamba features
    mamba_features = processor.extract_mamba_features(sample_packet)
    print(f"  - Mamba features shape: {mamba_features.shape}")
    print(f"  - Feature range: [{np.min(mamba_features):.3f}, {np.max(mamba_features):.3f}]")
    
    print("\n8. REWARD STATISTICS")
    print("-" * 20)
    
    stats = calculator.get_reward_statistics()
    coverage_trend = calculator.get_coverage_trend()
    
    print(f"✓ Reward statistics:")
    print(f"  - Mean reward: {stats['mean']:.3f}")
    print(f"  - Std deviation: {stats['std']:.3f}")
    print(f"  - Reward count: {stats['count']}")
    
    print(f"✓ Coverage trend:")
    print(f"  - Recent average: {coverage_trend['recent_average']:.3f}")
    print(f"  - Total coverage: {coverage_trend['total_coverage']:.3f}")
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\nKEY ACHIEVEMENTS:")
    print("✓ 90/10 field selection strategy implemented")
    print("✓ 6 mutation actions (insert, delete, flip, replace, shuffle, copy)")
    print("✓ Three-tier reward system (basic, medium, high)")
    print("✓ Mamba feature integration support")
    print("✓ Field importance learning")
    print("✓ Protocol data processing")
    print("✓ Comprehensive reward tracking")
    
    print("\nFRAMEWORK READY FOR:")
    print("• Integration with actual Mamba feature extractor")
    print("• Connection to real protocol fuzzing targets")
    print("• TensorFlow-based neural network training")
    print("• Large-scale fuzzing campaigns")


if __name__ == '__main__':
    demonstrate_fuzzing_framework()