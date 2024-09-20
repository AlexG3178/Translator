import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import BertTokenizer
from tqdm import tqdm

from models.transformer import Transformer
from data.dataset import get_dataloader_from_dataset
from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils.translate_utils import translate_sentence
from config import config


def main():    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ds = load_dataset("bstraehle/en-to-es-auto-finance", split="train")
    split_ds = ds.train_test_split(test_size=0.1, seed=41)  
    train_ds = split_ds['train']
    val_ds = split_ds['test']
   
    src_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Tokenizer for English
    tgt_tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')  # Tokenizer for Spanish
    
    src_vocab_size = src_tokenizer.vocab_size
    tgt_vocab_size = tgt_tokenizer.vocab_size

    transformer = Transformer(src_vocab_size, tgt_vocab_size, config['d_model'], config['num_heads'], config['num_layers'], config['d_ff'], config['dropout'])
    transformer.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=config['learning_rate'], betas=(0.9, 0.98), eps=1e-9, weight_decay=config['weight_decay'])
    
    # Define scheduler - StepLR reduces the learning rate by `gamma` every `step_size` epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    
    # Load checkpoint if exists
    start_epoch, best_val_loss = load_checkpoint(transformer, optimizer, scheduler, file_path="transformer_checkpoint.pth")

    transformer.train()
    
    dataloader = get_dataloader_from_dataset(train_ds, src_tokenizer, tgt_tokenizer, batch_size=32, max_length=128)

    best_loss = float('inf')
    train_losses = []
    val_losses = []
    val_accs = []
    epochs = []
    
    for epoch in range(start_epoch, config['num_epochs']):
        transformer.train()  # Training mode
        running_loss = 0
        for src_batch, tgt_batch in tqdm(dataloader):
            optimizer.zero_grad()
            
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            output = transformer(src_batch, tgt_batch[:, :-1])
            loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_batch[:, 1:].contiguous().view(-1))
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
             # Store training loss
        train_losses.append(running_loss / len(dataloader))
    
        transformer.eval()  # Evaluation mode
        val_loss_total = 0
        val_correct_total = 0
        
        val_dataloader = get_dataloader_from_dataset(val_ds, src_tokenizer, tgt_tokenizer, batch_size=config['batch_size'])

        with torch.no_grad():
            val_loss_total = 0
            val_correct_total = 0
            for val_src, val_tgt in val_dataloader:
                val_src = val_src.to(device)
                val_tgt = val_tgt.to(device)

                val_output = transformer(val_src, val_tgt[:, :-1])
                val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt[:, 1:].contiguous().view(-1))

                val_loss_total += val_loss.item()  # No need to move to CPU, val_loss.item() is already a Python float

                # Calculate accuracy and move to CPU before using it
                correct = (torch.argmax(val_output, dim=-1) == val_tgt[:, 1:]).float()
                val_correct_total += correct.sum().cpu() / correct.numel()  # Move to CPU before summing

        val_losses.append(val_loss_total / len(val_dataloader))
        val_accs.append((val_correct_total / len(val_dataloader)).item())  # Move to CPU and convert to scalar
        epochs.append(epoch + 1)
        
        print(f"Epoch: {epoch + 1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")
        
        scheduler.step()  # Learning rate adjustment after each epoch    
            
        # Check if validation loss improved
        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
            epochs_without_improvement = 0
            save_checkpoint(transformer, optimizer, scheduler, epoch + 1, val_losses[-1], file_path="transformer_checkpoint.pth")
        else:
            epochs_without_improvement += 1
            print(f"Validation loss did not improve. Epochs without improvement: {epochs_without_improvement}/{config['patience']}")
            
            # Early stopping check
            if epochs_without_improvement >= config['patience']:
                print(f"Early stopping at epoch {epoch + 1}")
                break 
        
    # Plotting the training history
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1) # Add an Axes to the current figure or retrieve an existing Axes
    plt.plot(epochs, train_losses, '-o', label='Train loss')
    plt.plot(epochs, val_losses, '-o', label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accs, '-o', label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()
    
    
    # Translate
    translated_sentence = translate_sentence(transformer, config['src_sentence'], src_tokenizer, tgt_tokenizer, device)
    print(f"Source: {config['src_sentence']}")
    print(f"Translated: {translated_sentence}")


if __name__ == "__main__":
    main()
