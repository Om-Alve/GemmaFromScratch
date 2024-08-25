# Gemma PyTorch Implementation

This repository provides a pure PyTorch implementation of the Gemma model. It includes functionalities for loading weights, running inference, and implementing key features such as KVCache, RMSNorm, RotaryPositionEmbeddings, and Grouped Multi Query Attention.

## Getting Started

### Clone the Gemma Repository

First, clone the Hugging Face `gemma` repository:

```bash
git lfs install
git clone https://huggingface.co/google/gemma-2b
```

### Installation

Clone this repo and navigate to it:

```bash
git clone https://github.com/Om-Alve/GemmaFromScratch.git
cd GemmaFromScratch
```

### Usage

```bash
python inference.py --prompt "Your input prompt here" --max_new_tokens 100 --temperature 1.0 --do_sample
```

Replace `"Your input prompt here"` with your input text. Adjust `max_new_tokens`, `temperature`, and `do_sample` as needed.

### Features Implemented

- **KVCache**: Efficiently manages key-value caches for attention mechanisms.
- **RMSNorm**: Implements Root Mean Square Layer Normalization.
- **RotaryPositionEmbeddings**: Applies rotary position embeddings.
- **Grouped Multi Query Attention**: Optimizes multi-query attention mechanisms.

### Example

To generate text using the Gemma model, use the following command:

```bash
python inference.py --prompt "Once upon a time" --max_new_tokens 50 --temperature 0.7 --do_sample
```

This will generate a continuation of the prompt "Once upon a time" with a maximum of 50 new tokens and a temperature of 0.7.


## Acknowledgements

This project is based on the Hugging Face `transformers` repository and is made with reference to their implementation of the Gemma model.