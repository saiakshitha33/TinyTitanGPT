# test_tinytitan.py

import torch
from tinytitan_gpt import TinyTitanGPT, build_vocab

# 🔸 Simulated mini dataset
text = "### Input:\nPt. has fever.\n\n### Output:\nSubjective: Fever."
chars, stoi, itos, encode, decode = build_vocab(text)
vocab_size = len(chars)

def test_tokenizer():
    s = "Pt. has fever."
    encoded_seq = encode(s)
    decoded_seq = decode(encoded_seq)
    assert decoded_seq == s, f"Decode failed: got '{decoded_seq}', expected '{s}'"
    print("✅ Tokenizer test passed!")

def test_model_forward():
    # Init model with vocab size
    model = TinyTitanGPT(vocab_size).to("cpu")

    # Dummy input for forward pass
    dummy_input = torch.randint(0, vocab_size, (2, 8))  # (batch, seq_len)
    logits, loss = model(dummy_input, dummy_input)

    # Check shapes
    assert logits.shape == (16, vocab_size), f"Expected (16, {vocab_size}), got {logits.shape}"
    assert loss.item() > 0, "Loss should be > 0"
    print("✅ Model forward pass test passed!")


if __name__ == "__main__":
    test_tokenizer()
    test_model_forward()
