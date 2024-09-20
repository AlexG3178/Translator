config = {
    'num_epochs': 5, # Early stopping: Epoch: 41, Train Loss: 0.0311, Val Loss: 0.3016, Val Acc: 0.9589
    'num_heads': 4, # was 8
    'num_layers': 4, # was6
    'd_model': 512,
    'd_ff': 2048,
    'dropout': 0.2,
    'weight_decay': 1e-4,
    'batch_size': 32,
    'learning_rate': 0.0001,
    'patience': 5,
    'epochs_without_improvement': 0,
    'src_sentence': "I'm not sure how long I want to finance my car for."
}