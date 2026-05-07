# Quick verification script to check setup
"""Verify that the RL stack is correctly set up."""

import sys
from pathlib import Path

def check_structure():
    """Check if all required files exist."""
    print("="*50)
    print("RL Stack Structure Verification")
    print("="*50)
    
    required_files = [
        # Core scripts
        'run_experiment.py',
        'evaluate.py',
        'run_all.sh',
        
        # Configs
        'configs/sac_config.yaml',
        'configs/dreamer_config.yaml',
        'configs/hierarchical_config.yaml',
        
        # Environment
        'src/envs/__init__.py',
        'src/envs/biped_env.py',
        
        # SAC
        'src/algorithms/sac/__init__.py',
        'src/algorithms/sac/sac.py',
        'src/algorithms/sac/networks.py',
        'src/algorithms/sac/replay_buffer.py',
        
        # Dreamer
        'src/algorithms/dreamer/__init__.py',
        'src/algorithms/dreamer/dreamer.py',
        'src/algorithms/dreamer/world_model.py',
        'src/algorithms/dreamer/actor_critic.py',
        
        # Hierarchical
        'src/algorithms/hierarchical/__init__.py',
        'src/algorithms/hierarchical/hierarchical.py',
        'src/algorithms/hierarchical/skill_manager.py',
        'src/algorithms/hierarchical/low_level_controller.py',
        
        # Utils
        'src/utils/__init__.py',
        'src/utils/seeding.py',
        'src/utils/logging.py',
        'src/utils/checkpointing.py',
        'src/utils/config.py',
        'src/utils/video.py',
        
        # Tests
        'tests/test_rl_stack.py',
        
        # Documentation
        'README.md',
        'EXECUTION_GUIDE.md',
        'reproducibility.md',
        'citations.md',
        'DONE.md',
        
        # Dependencies
        'requirements.txt',
        'environment.yml',
        'Dockerfile',
    ]
    
    missing = []
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing.append(file)
    
    print("="*50)
    if missing:
        print(f"❌ Missing {len(missing)} files:")
        for f in missing:
            print(f"  - {f}")
        return False
    else:
        print("✅ All required files present!")
        return True


def check_directories():
    """Check if required directories exist."""
    print("\n" + "="*50)
    print("Directory Structure Verification")
    print("="*50)
    
    required_dirs = [
        'src',
        'src/envs',
        'src/algorithms',
        'src/algorithms/sac',
        'src/algorithms/dreamer',
        'src/algorithms/hierarchical',
        'src/utils',
        'configs',
        'tests',
        'results',
        'results/checkpoints',
        'results/tensorboard',
        'results/videos',
        'results/evaluations',
        'docs',
    ]
    
    missing = []
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/")
            missing.append(dir_path)
    
    print("="*50)
    if missing:
        print(f"⚠️  Missing {len(missing)} directories (will be created as needed)")
        return True  # Not critical
    else:
        print("✅ All directories present!")
        return True


def main():
    """Main verification routine."""
    print("\n🚀 RL Bipedal Robot Stack Verification\n")
    
    files_ok = check_structure()
    dirs_ok = check_directories()
    
    print("\n" + "="*50)
    print("VERIFICATION SUMMARY")
    print("="*50)
    
    if files_ok and dirs_ok:
        print("✅ Setup complete! Ready to train.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run tests: pytest tests/ -v")
        print("  3. Start training: python run_experiment.py --config configs/sac_config.yaml")
        print("  4. Or run all: ./run_all.sh")
        return 0
    else:
        print("❌ Setup incomplete. Please check missing files.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
