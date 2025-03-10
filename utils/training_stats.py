# utils/training_stats.py

def print_training_stats(epoch_times, model):
    """
    Prints training statistics including total training time, average epoch time,
    and model parameter counts.

    Args:
        epoch_times (list[float]): A list containing the time taken for each epoch (in seconds).
        model (torch.nn.Module): The model whose parameters are counted.
    """
    total_time = sum(epoch_times)
    avg_epoch_time = total_time / len(epoch_times) if epoch_times else 0

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=== Training Statistics ===")
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Average epoch time: {avg_epoch_time:.2f} seconds")
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
