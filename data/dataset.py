from torch.utils.data import Dataset, DataLoader


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
    
    src_sentences = dataset['en']
    tgt_sentences = dataset['es']
    dataset = TranslationDataset(src_sentences, tgt_sentences, src_tokenizer, tgt_tokenizer, max_length)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
