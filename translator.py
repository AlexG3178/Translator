import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BertTokenizer
from tqdm import tqdm


class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_tokenizer, tgt_tokenizer, max_length=100):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]

        # Add special tokens
        src_sentence = "<SOS> " + src_sentence + " <EOS>"
        tgt_sentence = "<SOS> " + tgt_sentence + " <EOS>"

        # Tokenize both sentences
        src_tokens = self.src_tokenizer.encode(src_sentence, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt").squeeze()
        tgt_tokens = self.tgt_tokenizer.encode(tgt_sentence, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt").squeeze()

        return src_tokens, tgt_tokens
    
    
def get_dataloader_from_dataset(dataset, src_tokenizer, tgt_tokenizer, batch_size=64, max_length=100):
    
    src_sentences = [example['translation']['en'] for example in dataset['train']]
    tgt_sentences = [example['translation']['es'] for example in dataset['train']]
    dataset = TranslationDataset(src_sentences, tgt_sentences, src_tokenizer, tgt_tokenizer, max_length)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=512):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        seq_length = x.size(1)
        return x + self.pe[:, :seq_length]
    
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=src.device), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
    
    
#region Helper Functions

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
        checkpoint = torch.load(file_path)
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

#endregion


    
#region Main

def main():    

    num_epochs = 50
    num_heads = 8
    num_layers = 6
    d_model = 512
    d_ff = 2048
    dropout = 0.1
    batch_size = 32
    learning_rate = 0.0001
    
    # Automatically use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    src_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Tokenizer for English
    tgt_tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')  # Tokenizer for Spanish
    
    src_vocab_size = src_tokenizer.vocab_size
    tgt_vocab_size = tgt_tokenizer.vocab_size

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, dropout)
    transformer.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    
    # Define scheduler - StepLR reduces the learning rate by `gamma` every `step_size` epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    
    # Load checkpoint if exists
    start_epoch, best_val_loss = load_checkpoint(transformer, optimizer, scheduler, file_path="transformer_checkpoint.pth")

    transformer.train()
    
    ds = load_dataset("qanastek/WMT-16-PubMed", "en-es")
    dataloader = get_dataloader_from_dataset(ds, src_tokenizer, tgt_tokenizer, batch_size=32, max_length=128)

    for epoch in range(start_epoch, num_epochs):
        for src_batch, tgt_batch in tqdm(dataloader):
            optimizer.zero_grad()
            
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            output = transformer(src_batch, tgt_batch[:, :-1])
            loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_batch[:, 1:].contiguous().view(-1))
            
            loss.backward()
            optimizer.step()
            scheduler.step() # Update learning rate
            
        print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
   
    # Save checkpoint every 10 epochs or if validation loss improves
        if (epoch + 1) % 10 == 0 or loss.item() < best_val_loss:
            save_checkpoint(transformer, optimizer, scheduler, epoch + 1, loss.item(), file_path="transformer_checkpoint.pth")
            best_val_loss = min(best_val_loss, loss.item())
        
    transformer.eval()
    
     # Validation set (not included in dataset download above, you'll need to adjust for this)
    val_ds = load_dataset("qanastek/WMT-16-PubMed", "en-es", split="validation")
    val_dataloader = get_dataloader_from_dataset(val_ds, src_tokenizer, tgt_tokenizer, batch_size=batch_size)

    with torch.no_grad(): 
        val_loss_total = 0
        for val_src, val_tgt in val_dataloader:
            val_output = transformer(val_src, val_tgt[:, :-1])
            val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt[:, 1:].contiguous().view(-1))
            val_loss_total += val_loss.item()
        
        print(f"Validation Loss: {val_loss_total / len(val_dataloader):.4f}")
        
        # Save final model
        save_checkpoint(transformer, optimizer, scheduler, num_epochs, val_loss.item(), file_path="transformer_final.pth")
        

if __name__ == "__main__":
    main()
    
#endregion