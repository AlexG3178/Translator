import torch


def translate_sentence(model, src_sentence, src_tokenizer, tgt_tokenizer, device, max_length=100):
    model.eval()  
    
    # Tokenize the source sentence
    src_tokens = src_tokenizer.encode("<SOS> " + src_sentence + " <EOS>", padding='max_length', truncation=True, max_length=max_length, return_tensors="pt").to(device)
    # print(f"Source tokens: {src_tokens}")  
    
    # Create an empty target sequence to store the translation output
    tgt_tokens = torch.zeros((1, max_length), dtype=torch.long).to(device)
    tgt_tokens[0, 0] = tgt_tokenizer.convert_tokens_to_ids("<SOS>")  # Start with <SOS>

    for i in range(1, max_length):
        output = model(src_tokens, tgt_tokens[:, :i])  # Pass source and current target tokens

        next_token_logits = output[:, -1, :]  # Get the logits for the last token in the sequence
        next_token_id = torch.argmax(next_token_logits, dim=-1).item()  # Find the predicted token
        
        # print(f"Generated token {i}: {tgt_tokenizer.decode([next_token_id])}") 

        tgt_tokens[0, i] = next_token_id  # Add predicted token to the target sequence

        # Stop when we hit the end token
        if next_token_id == tgt_tokenizer.convert_tokens_to_ids("<EOS>"):
            # print("EOS token generated. Ending translation.")
            break

    # Decode the token IDs back into words
    tgt_sentence = tgt_tokenizer.decode(tgt_tokens[0], skip_special_tokens=True)
    # print(f"Final target tokens: {tgt_tokens}") 
    return tgt_sentence
