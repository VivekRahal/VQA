# utils/gpu_utils.py
import torch
import time
import gc

def check_gpu_availability():
    """Check GPU availability and print detailed information."""
    print("\n=== GPU Information ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Check current memory usage
        allocated = torch.cuda.memory_allocated(0) / 1e9
        cached = torch.cuda.memory_reserved(0) / 1e9
        print(f"GPU Memory Allocated: {allocated:.2f} GB")
        print(f"GPU Memory Cached: {cached:.2f} GB")
        
        return True
    else:
        print("❌ No GPU available - using CPU")
        return False

def optimize_gpu_memory():
    """Optimize GPU memory usage."""
    if torch.cuda.is_available():
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ GPU memory optimized")

def transfer_to_gpu(batch_data, device, verbose=False):
    """
    Efficiently transfer batch data to GPU with memory optimization.
    
    Args:
        batch_data: Tuple of (images, questions, answers) or similar
        device: Target device (cuda or cpu)
        verbose: Print transfer information
    
    Returns:
        Tuple of tensors on target device
    """
    if verbose:
        print(f"Transferring batch to {device}")
        if torch.cuda.is_available():
            print(f"GPU Memory before transfer: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    start_time = time.time()
    
    # Transfer each tensor to device
    transferred_data = []
    for tensor in batch_data:
        if isinstance(tensor, torch.Tensor):
            transferred_tensor = tensor.to(device, non_blocking=True)
            transferred_data.append(transferred_tensor)
        else:
            transferred_data.append(tensor)
    
    transfer_time = time.time() - start_time
    
    if verbose:
        print(f"Transfer completed in {transfer_time:.4f}s")
        if torch.cuda.is_available():
            print(f"GPU Memory after transfer: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    return tuple(transferred_data)

def monitor_gpu_usage(func):
    """Decorator to monitor GPU usage during function execution."""
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            print(f"\n=== GPU Usage Monitor ===")
            print(f"Before {func.__name__}: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            print(f"After {func.__name__}: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"Execution time: {end_time - start_time:.2f}s")
            print("=" * 30)
            
            return result
        else:
            return func(*args, **kwargs)
    
    return wrapper

def get_optimal_batch_size(model, sample_input, max_memory_gb=3.5):
    """
    Determine optimal batch size based on GPU memory.
    
    Args:
        model: PyTorch model
        sample_input: Sample input tensor
        max_memory_gb: Maximum GPU memory to use (default 3.5GB for 4GB GPU)
    
    Returns:
        Optimal batch size
    """
    if not torch.cuda.is_available():
        return 2  # Default for CPU
    
    device = torch.device("cuda")
    model = model.to(device)
    
    # Start with small batch size
    batch_size = 1
    max_batch_size = 32
    
    while batch_size <= max_batch_size:
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Create batch
            batch_input = sample_input.repeat(batch_size, 1, 1, 1) if len(sample_input.shape) == 4 else sample_input.repeat(batch_size, 1)
            
            # Move to GPU
            batch_input = batch_input.to(device)
            
            # Forward pass
            with torch.no_grad():
                _ = model(batch_input)
            
            # Check memory usage
            memory_used = torch.cuda.memory_allocated(0) / 1e9
            
            if memory_used > max_memory_gb:
                batch_size -= 1
                break
            
            batch_size += 1
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                batch_size -= 1
                break
            else:
                raise e
    
    # Clean up
    torch.cuda.empty_cache()
    
    print(f"Optimal batch size: {batch_size}")
    return max(1, batch_size)

def print_gpu_memory_summary():
    """Print current GPU memory summary."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        cached = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"\nGPU Memory Summary:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Cached: {cached:.2f} GB")
        print(f"  Total: {total:.2f} GB")
        print(f"  Usage: {(allocated/total)*100:.1f}%")

if __name__ == "__main__":
    check_gpu_availability()
    print_gpu_memory_summary() 