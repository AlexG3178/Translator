import torch
import os


def save_checkpoint(model, optimizer, scheduler, epoch, loss, file_path="transformer_checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at {file_path}")

def load_checkpoint(model, optimizer, scheduler, file_path="transformer_checkpoint.pth"):
    if os.path.isfile(file_path):
        checkpoint = torch.load(file_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded from {file_path} (epoch {epoch}, loss {loss:.4f})")
        return epoch, loss
    else:
        print(f"No checkpoint found at {file_path}")
        return 0, float('inf')  # If no checkpoint, start from epoch 0