import torch
from utils import load_model
from gemma import KVCache, GemmaForCausalLM
from transformers import AutoTokenizer


def move_to_device(inputs: dict, device: str) -> dict:
    return {k: v.to(device) for k, v in inputs.items()}


def min_p_sampling(logits: torch.Tensor, p_base: float = 0.1) -> torch.Tensor:
    p_max = logits.max(dim=-1, keepdim=True).values
    p_scaled = p_max * p_base
    mask = logits >= p_scaled
    logits = logits * mask.float()
    logits = logits / logits.sum(dim=-1, keepdim=True)
    next_token = torch.multinomial(logits, num_samples=1)
    return next_token.squeeze(-1)


def inference(
    model: GemmaForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    device: str,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs["position_ids"] = inputs["attention_mask"].cumsum(dim=-1)
    inputs = move_to_device(inputs, device)

    kv_cache = KVCache()
    stop_token = tokenizer.eos_token_id
    generated_tokens = []

    for _ in range(max_new_tokens):
        outputs = model(**inputs, kv_cache=kv_cache)
        kv_cache = outputs["kv_cache"]
        logits = outputs["logits"][:, -1, :]

        if do_sample:
            logits = torch.softmax(logits / temperature, dim=-1)
            next_token = min_p_sampling(logits)
        else:
            next_token = logits.argmax(dim=-1)

        generated_tokens.append(next_token)
        if next_token.item() == stop_token:
            break

        inputs["input_ids"] = next_token.unsqueeze(-1)
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones((1, 1), device=device)], dim=-1
        )
        inputs["position_ids"] = (
            inputs["attention_mask"].cumsum(dim=-1)[:, -1].unsqueeze(0)
        )

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=False)

    return prompt + decoded


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "../gemma-2b"
    print(f"Loading model")
    model, tokenizer = load_model(model_path, device)
    model = model.to(device).eval()

    prompt = "1 + 1 ="
    max_new_tokens = 10
    temperature = 1.0
    do_sample = True

    print("Model Loaded!")

    with torch.no_grad():
        output = inference(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            device=device,
        )
    print(output)
