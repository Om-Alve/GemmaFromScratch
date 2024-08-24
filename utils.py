from transformers import AutoTokenizer
import glob
from safetensors import safe_open
import json
from typing import Tuple
import os
import torch
from gemma import GemmaConfig, GemmaForCausalLM, KVCache


def load_model(model_path: str, device: str) -> Tuple[GemmaForCausalLM, AutoTokenizer]:

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")

    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    tensors = {}

    for safetensor_file in safetensor_files:
        with safe_open(safetensor_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = GemmaConfig(**model_config_file)

    model = GemmaForCausalLM(config)

    model.load_state_dict(tensors, strict=False)
    model.tie_weights()
    return (model, tokenizer)


if __name__ == "__main__":

    model_path = "../gemma-2-2b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(model_path, device)
    print("Model Loaded Successfully!")
    inputs = tokenizer("Hello world", return_tensors='pt')
    kv_cache = KVCache()
    inputs['position_ids'] = torch.cumsum(inputs['attention_mask'],dim=-1) - 1
    print(inputs)
    out = model(**inputs, kv_cache=kv_cache)
    kv_cache = out['kv_cache']
    logits = out['logits']
    print(logits.shape)
    logits = logits.argmax(dim=-1)
    print(logits)
    # print(tokenizer.decode(logits))

