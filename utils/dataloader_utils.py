# utils/dataloader_utils.py
import os
import torch
import multiprocessing
import platform

def get_optimal_workers():
    """
    Determine optimal number of workers for DataLoader based on system.
    
    Returns:
        int: Recommended number of workers
    """
    # Get CPU count
    cpu_count = multiprocessing.cpu_count()
    
    # Platform-specific recommendations
    if platform.system() == "Windows":
        # Windows has issues with multiprocessing, use fewer workers
        if cpu_count <= 4:
            return 0  # Use main process only
        elif cpu_count <= 8:
            return 2
        else:
            return 4
    else:
        # Linux/Mac can handle more workers
        if cpu_count <= 4:
            return 2
        elif cpu_count <= 8:
            return 4
        else:
            return min(8, cpu_count // 2)

def get_dataloader_config():
    """
    Get recommended DataLoader configuration for current system.
    
    Returns:
        dict: Configuration dictionary with recommended settings
    """
    num_workers = get_optimal_workers()
    
    config = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 2 if num_workers > 0 else None
    }
    
    return config

def print_system_info():
    """Print system information for DataLoader optimization."""
    print("\n=== System Information for DataLoader Optimization ===")
    print(f"Platform: {platform.system()}")
    print(f"CPU Cores: {multiprocessing.cpu_count()}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    config = get_dataloader_config()
    print(f"\nRecommended DataLoader Settings:")
    print(f"  num_workers: {config['num_workers']}")
    print(f"  pin_memory: {config['pin_memory']}")
    print(f"  persistent_workers: {config['persistent_workers']}")
    if config['prefetch_factor']:
        print(f"  prefetch_factor: {config['prefetch_factor']}")
    
    print("\nPerformance Tips:")
    if config['num_workers'] == 0:
        print("  ⚠️  Using 0 workers (main process only)")
        print("  💡 Consider increasing if you have more CPU cores")
    else:
        print(f"  ✅ Using {config['num_workers']} workers for parallel loading")
    
    if config['pin_memory']:
        print("  ✅ Pin memory enabled for faster GPU transfer")
    else:
        print("  ℹ️  Pin memory disabled (no GPU detected)")
    
    if config['persistent_workers']:
        print("  ✅ Persistent workers enabled for better performance")
    else:
        print("  ℹ️  Persistent workers disabled (no workers)")
    
    print("=" * 50)

def update_config_with_optimal_settings(config):
    """
    Update a configuration object with optimal DataLoader settings.
    
    Args:
        config: Configuration object (Config or ModularConfig)
    
    Returns:
        config: Updated configuration object
    """
    dataloader_config = get_dataloader_config()
    
    config.num_workers = dataloader_config["num_workers"]
    config.pin_memory = dataloader_config["pin_memory"]
    config.persistent_workers = dataloader_config["persistent_workers"]
    
    print(f"Updated config with optimal DataLoader settings:")
    print(f"  num_workers: {config.num_workers}")
    print(f"  pin_memory: {config.pin_memory}")
    print(f"  persistent_workers: {config.persistent_workers}")
    
    return config

if __name__ == "__main__":
    print_system_info() 