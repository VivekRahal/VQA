# example_optimal_dataloader.py
"""
Example script showing how to use optimal DataLoader settings.
This script demonstrates different ways to configure DataLoader workers.
"""

from config import Config
from modular_config import ModularConfig
from utils.dataloader_utils import print_system_info, update_config_with_optimal_settings, get_optimal_workers

def example_original_system():
    """Example for original system with optimal DataLoader settings."""
    print("\n=== Original System Example ===")
    
    # Method 1: Manual configuration
    config = Config()
    config.num_workers = 4  # Manual setting
    config.pin_memory = True
    config.persistent_workers = True
    print(f"Manual config - num_workers: {config.num_workers}")
    
    # Method 2: Automatic optimal configuration
    config_auto = Config()
    config_auto = update_config_with_optimal_settings(config_auto)
    print(f"Auto config - num_workers: {config_auto.num_workers}")

def example_modular_system():
    """Example for modular system with optimal DataLoader settings."""
    print("\n=== Modular System Example ===")
    
    # Method 1: Manual configuration
    config = ModularConfig()
    config.num_workers = 2  # Manual setting
    config.pin_memory = True
    config.persistent_workers = True
    print(f"Manual config - num_workers: {config.num_workers}")
    
    # Method 2: Automatic optimal configuration
    config_auto = ModularConfig()
    config_auto = update_config_with_optimal_settings(config_auto)
    print(f"Auto config - num_workers: {config_auto.num_workers}")
    
    # Method 3: Custom preset with optimal workers
    config_preset = ModularConfig()
    config_preset.get_preset_config("vit_bert_coattention")
    config_preset = update_config_with_optimal_settings(config_preset)
    print(f"Preset config - num_workers: {config_preset.num_workers}")

def example_performance_comparison():
    """Example showing performance impact of different worker settings."""
    print("\n=== Performance Comparison Example ===")
    
    # Test different worker configurations
    worker_configs = [0, 2, 4, 8]
    
    for num_workers in worker_configs:
        config = ModularConfig()
        config.num_workers = num_workers
        config.pin_memory = True
        config.persistent_workers = num_workers > 0
        
        print(f"\nWorkers: {num_workers}")
        if num_workers == 0:
            print("  - Single process loading")
            print("  - Slower but more stable")
        else:
            print(f"  - Parallel loading with {num_workers} workers")
            print("  - Faster but uses more memory")
            print(f"  - Persistent workers: {config.persistent_workers}")

def example_windows_specific():
    """Example for Windows-specific DataLoader configuration."""
    print("\n=== Windows-Specific Example ===")
    
    import platform
    if platform.system() == "Windows":
        print("Windows detected - using conservative settings")
        
        config = ModularConfig()
        config.num_workers = 0  # Start with 0 on Windows
        config.pin_memory = True
        config.persistent_workers = False
        
        print("Recommended Windows settings:")
        print(f"  num_workers: {config.num_workers}")
        print(f"  pin_memory: {config.pin_memory}")
        print(f"  persistent_workers: {config.persistent_workers}")
        
        print("\nIf you have a powerful Windows machine, you can try:")
        print("  config.num_workers = 2  # Conservative")
        print("  config.num_workers = 4  # Aggressive")
    else:
        print("Non-Windows system - can use more workers")

def main():
    """Main function demonstrating DataLoader optimization."""
    print("🚀 DataLoader Optimization Examples")
    print("=" * 50)
    
    # Show system information
    print_system_info()
    
    # Examples
    example_original_system()
    example_modular_system()
    example_performance_comparison()
    example_windows_specific()
    
    print("\n" + "=" * 50)
    print("💡 Tips for optimal DataLoader performance:")
    print("1. Start with 0 workers and gradually increase")
    print("2. Monitor memory usage when increasing workers")
    print("3. Use persistent_workers=True for better performance")
    print("4. Enable pin_memory=True if using GPU")
    print("5. Windows users should be more conservative with workers")
    print("6. Test different settings with your specific dataset")

if __name__ == "__main__":
    main() 