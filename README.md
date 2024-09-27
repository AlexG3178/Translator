Transformer-Based Machine Translation
This project implements a Transformer-based machine translation model using PyTorch. 
The model using the Transformer architecture, featuring multi-head self-attention, positional encodings, and multiple encoder-decoder layers. 

Table of Contents:
1. Overview
2. Requirements
3. Dataset
4. Model Architecture
5. Setup
6. Training
7. Evaluation
8. Checkpointing
9. Visualization

Overview
This project is a machine translation system based on the original Transformer model proposed in "Attention is All You Need". 
The model is trained to translate English sentences into Spanish using a financial dataset.

Key features:
- Transformer architecture with multi-head self-attention
- Positional encodings
- Cross-attention between encoder and decoder
- Custom learning rate scheduler
- Checkpointing for model saving and loading

Requirements
Ensure you have the following packages installed:
- Python 3.7+
- PyTorch 1.9.0+
- Hugging Face transformers library
- datasets library
- matplotlib for plotting
- tqdm for progress bars
- CUDA (optional, for GPU training)

Dataset
The dataset used for training is an English-to-Spanish translation dataset from the financial domain:
Source: Hugging Face Dataset
Split: 90% for training and 10% for validation.
We tokenize the English and Spanish sentences using BERT-based tokenizers.

Model Architecture
The model follows the Transformer architecture with:
- Encoder: Processes the source (English) sentence.
 -Decoder: Generates the target (Spanish) sentence, conditioned on the encoder's output.
 -Positional Encoding: Adds sequential information to input embeddings.
- Multi-head Attention: Allows the model to focus on different parts of the sentence simultaneously.

Setup
Clone the repository:
git clone https://github.com/yourusername/transformer-translation.git
cd transformer-translation
Install the dependencies:
- pip install 
(Optional) If you plan to train the model on a GPU, ensure you have CUDA installed and PyTorch is configured to use CUDA.

Training
To train the model, run the main() function in translator.py. This function handles:
- Dataset loading
- Tokenization (using BERT-based tokenizers)
- Training the model over multiple epochs
- Saving the model checkpoints

Sample Output During Training
Epoch: 1, Train Loss: 3.4567, Val Loss: 2.9876, Val Acc: 65.23%
Epoch: 2, Train Loss: 2.1234, Val Loss: 1.9876, Val Acc: 72.14%

Evaluation
Once the model is trained, the validation set is used to evaluate its performance. 
The validation loss and accuracy are printed after every epoch. 
You can modify the validation set or test it on a new dataset by changing the dataset split inside the script.

Checkpointing
The model's state is saved after every 10 epochs or when the validation loss improves. 
You can resume training or load the final model by using the load_checkpoint() function. 
Checkpoints are saved in the file transformer_checkpoint.pth.

Visualization
The script generates training and validation loss plots after the training loop. 
These are displayed using matplotlib. You can compare how the model performs over different epochs by inspecting the loss curves.
