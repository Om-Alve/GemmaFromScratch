import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
import math

class GemmaConfig:
    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        intermediate_size: int,
        head_dim: int,
        rms_norm_eps: float,
        attention_bias: bool,
        rope_theta: int,
        attention_dropout: bool,
        pad_token_id: int,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: int):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        self.inv_freq.to(x.device)

        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)

            emb = torch.cat((freqs, freqs), dim=-1)

            cos = emb.cos()
            sin = emb.sin()

        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)

    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class KVCache:
    def __init__(self):
        self.key_cache = []
        self.value_cache = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            return self.key_cache[0].shape[-2]

    def update(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(keys)
            self.value_cache.append(values)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], keys], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], values], dim=-2
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


def repeat_kv(hidden_states: torch.Tensor, num_repeat: int):
    batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if num_repeat == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_repeat, seq_len, head_dim
    )

    return hidden_states.reshape(batch_size, num_key_value_heads * num_repeat, seq_len, head_dim)


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads 
        self.is_causal = True

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.rotary_embeddings = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: KVCache,
    ):
        batch_size, seq_len, _ = hidden_states.shape

        q = (
            self.q_proj(hidden_states)
            .view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        cos, sin = self.rotary_embeddings(v, position_ids)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / (self.head_dim**0.5)

        attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.config.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(batch_size, seq_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim_size: int, rms_norm_eps: float):
        super().__init__()
        self.dim_size = dim_size
        self.rms_norm_eps = rms_norm_eps
        self.weight = nn.Parameter(torch.zeros(self.dim_size))

    def _normalize(self, x: torch.Tensor):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.rms_norm_eps)

    def forward(self, x: torch.Tensor):

        x = self._normalize(x.float())

        x = x * (1.0 + self.weight.float())

        return x


class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config

        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor):
        out = nn.functional.gelu(
            self.gate_proj(hidden_states), approximate="tanh"
        ) * self.up_proj(hidden_states)
        out = self.down_proj(out)
        return out


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.self_attn = GemmaAttention(config, layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size, rms_norm_eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, rms_norm_eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: KVCache,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = GemmaRMSNorm(config.hidden_size, rms_norm_eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * (self.config.hidden_size ** 0.5)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        dtype, device = hidden_states.dtype, hidden_states.device
        q_len = hidden_states.shape[1]
        
        if kv_cache is None or kv_cache.num_items() == 0:
            causal_mask = torch.triu(torch.full((q_len, q_len), float('-inf'), device=device), diagonal=1)
        else:
            assert q_len == 1, "q_len should be 1 when using KV cache"
            kv_len = kv_cache.num_items() + q_len
            causal_mask = torch.full((q_len, kv_len), 0, dtype=dtype, device=device)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask.unsqueeze(0).unsqueeze(0),
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states

class GemmaForCausalLM(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        logits = self.lm_head(outputs)
        logits = logits.float()

        return_data = {"logits": logits}
        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        return return_data