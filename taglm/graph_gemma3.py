import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

from typing import Optional, Tuple

class Cache:
    """
    Base class for all cache implementations.
    """
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement the `update` method.")


class DynamicCache(Cache):
    """
    A dynamic cache that grows as new tokens are generated.
    Stores key and value states for each layer.
    """
    def __init__(self) -> None:
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self._seen_tokens = 0  # Used in `generate` to track how many tokens have been processed

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Support indexing to retrieve cached states for a specific layer.
        Returns (key_states, value_states) for the given layer.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer {layer_idx}")

    def __iter__(self):
        """
        Support iteration over layers.
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Return the number of layers with cached states.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new key and value states for a given layer.
        
        Args:
            key_states: New key states to add [batch_size, num_heads, seq_len, head_dim]
            value_states: New value states to add [batch_size, num_heads, seq_len, head_dim]
            layer_idx: Index of the layer being updated
            cache_kwargs: Additional arguments (unused in basic implementation)
            
        Returns:
            Tuple of (updated_key_states, updated_value_states)
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            # First time we see this layer, initialize the cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Concatenate new states with existing cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length (None for dynamic cache)."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search (if needed)."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))




class SiglipVisionConfig():
    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=3072,
        image_size=4304,
        model_type="siglip_vision_model",
        num_attention_heads=16,
        num_hidden_layers=27,
        num_channels=3,
        patch_size=14,
        vision_use_head=False,
        attention_dropout=0.0,
        layer_norm_eps=1e-6,
        hidden_act="gelu_pytorch_tanh"
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.image_size = image_size
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.vision_use_head = vision_use_head
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        






class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)
        
        
    def interpolate_pos_encoding(self, embeddings, height, width):
        pass
    
    def forward(self, pixel_values, interpolate_pos_encoding=False):
        batch_size, num_channels, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype)) # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        
        return embeddings
        
        
        





class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = nn.GELU(approximate="tanh") # ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    

class SiglipAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got {self.embed_dim} and {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False
        
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        batch_size, seq_length, embed_dim = hidden_states.shape
        
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)
        
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(queries, keys.transpose(2, 3)) * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(queries.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_outputs = torch.matmul(attn_weights, values)
        
        attn_outputs = attn_outputs.transpose(1, 2).contiguous()
        attn_outputs = attn_outputs.view(batch_size, seq_length, embed_dim)
        attn_outputs = self.out_proj(attn_outputs)
        
        return attn_outputs, attn_weights


        
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config): # visionconfig + textconfig
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        
    
    def forward(
        self,
        hidden_states,
        attention_mask,
    ):
        residual = hidden_states
        
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states, attn_weights)
        
        return outputs







class SiglipEncoder(nn.Module):
    def __init__(self, config): # SiglipConfig
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        
    def forward(self,
                inputs_embeds,
                attention_mask=None
    ):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            
            layer_outputs = encoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )
            
            hidden_states = layer_outputs[0]
            
        return hidden_states
        


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hiden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        
    def forward(self, hidden_state):
        batch_size = hidden_state[0]
        probe = self.probe.repeat(batch_size, 1, 1)
        
        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]
        
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        
        return hidden_state[:, 0]




class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(config)
    
    def forward(
        self,
        pixel_values,
    ):
        hidden_states = self.embeddings(pixel_values)
        
        encoder_outputs = self.encoder(hidden_states)
        
        last_hidden_states = encoder_outputs # encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_states)
        
        pooler_output = self.head(last_hidden_state) if self.use_head else None
        
        return pooler_output if self.use_head else last_hidden_state





class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
        
    def get_input_embeddings(self):
        return self.vision_model.embeddings.patch_embedding
        
    def forward(self, pixel_values):
        # [Batch_size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)
    



class Gemma3TextScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, embed_scale=1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)
        
    def forward(self, input_ids):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)



class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # self.act_fn = nn.GELU()
        
    
    def forward(self, x):
        gate = F.gelu(self.gate_proj(x), approximate='tanh')
        up = self.up_proj(x)
        down_proj = self.down_proj(gate*up)
        return down_proj
        
        


class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)
    
    
    

# =====================================================================

class Gemma3RotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.config = config
        
        # if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
        #     self.scaling_factor = config.rope_scaling['factor']


        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0
    
    @torch.no_grad()
    def forward(self, x, position_ids):
        
        # if hasattr(self.config, "rope_scaling") and self.config.rope_scaling['rope_type']=="linear":
        #     position_ids = position_ids.float() / self.scaling_factor
        
        
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# =====================================================================



def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
    


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



class Gemma3Attention(nn.Module):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = True
        
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.sliding_window = config.sliding_window if self.is_sliding else None
        
        self.q_norm = Gemma3RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(config.head_dim, config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache,
        cache_position,
    ):

        bsz, q_len, _ = hidden_states.shape
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if past_key_values is not None:
            # Retrieve cached key and value states, then update with new states
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )
        # Repeat KVs for Grouped-Query Attention
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # --- Attention Calculation ---
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        
        if attention_mask is not None:
            # The boolean mask from create_*_mask needs to be converted to a float mask
            # float_mask = torch.where(attention_mask, torch.finfo(query_states.dtype).min)
            # attn_weights = attn_weights + float_mask
            attn_weights = attn_weights.masked_fill(attention_mask, torch.finfo(attn_weights.dtype).min)
            
            
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)
        # --- End Attention Calculation ---

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights
            
            
            
class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Gemma3Attention(config, layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(self.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, config.rms_norm_eps)
        
    def forward(
        self,
        hidden_states,
        position_embeddings_global,
        position_embeddings_local,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache,
        cache_position,
    ):
        
        
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        
        position_embeddings = position_embeddings_local if self.self_attn.is_sliding else position_embeddings_global
        
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            cache_position,
        )
        
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        outputs += (self_attn_weights,)
        
        return outputs


def create_causal_mask(
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    cache_position: torch.LongTensor,
    config=None,
    past_key_values=None,
    position_ids=None,
    graph_attention_mask=None,
    or_mask_function=None,
    graph_mask_mode='replace',  # 'add' or 'replace'
):
    batch_size, seq_len = input_embeds.shape[:2]
    key_len = cache_position[-1] + 1
    query_pos = cache_position.view(seq_len, 1)
    key_pos = torch.arange(key_len, device=input_embeds.device).view(1, key_len)
    causal_mask = query_pos < key_pos
    final_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    if attention_mask is not None and attention_mask.any():
        
        padding_mask = attention_mask[:, None, None, :] == 0 # Invert padding mask
        final_mask = final_mask | padding_mask

    # Apply or_mask_function for special token type handling (e.g., image tokens)
    if or_mask_function is not None:
        or_mask = torch.zeros_like(final_mask, dtype=torch.bool)
        for b in range(batch_size):
            for q in range(seq_len):
                for kv in range(key_len):
                    # Convert indices to tensors for the mask function
                    b_t = torch.tensor(b, device=input_embeds.device)
                    q_t = torch.tensor(q, device=input_embeds.device)
                    kv_t = torch.tensor(kv, device=input_embeds.device)
                    mask_val = or_mask_function(b_t, 0, q_t, kv_t)
                    if mask_val.item() if isinstance(mask_val, torch.Tensor) else mask_val:
                        or_mask[b, :, q, kv] = True
        # Where or_mask is True, we ALLOW attention (set final_mask to False)
        final_mask = final_mask & ~or_mask

    # Apply graph attention mask
    if graph_attention_mask is not None:
        # graph_attention_mask shape: [batch_size, orig_seq_len, orig_seq_len]
        # During generation, key_len may be larger than orig_seq_len
        # We need to handle this by only applying the mask to the overlapping region
        graph_seq_len = graph_attention_mask.shape[1]

        # Only apply graph mask to positions that exist in the graph mask
        if key_len <= graph_seq_len and seq_len <= graph_seq_len:
            # Normal case: mask fits the current sequence
            # Slice the graph mask to match current query and key positions
            graph_mask_slice = graph_attention_mask[:, :seq_len, :key_len]
            graph_mask_expanded = graph_mask_slice.unsqueeze(1)

            # PUNCH HOLE mode: Replace graph region with graph mask, keep causal mask for text
            # Convert graph_attention_mask to mask format (True->False=allow, False->True=block)
            graph_as_final_mask = ~graph_mask_expanded

            # Identify which positions are part of the graph region
            # A position is in graph region if it has any connections (row or column has True values)
            has_graph_connections = (graph_mask_expanded.any(dim=-1, keepdim=True) |
                                   graph_mask_expanded.any(dim=-2, keepdim=True))

            # Replace graph region with graph mask, keep causal mask elsewhere
            final_mask = torch.where(has_graph_connections, graph_as_final_mask, final_mask)

        elif seq_len <= graph_seq_len:
            # Key length exceeds graph mask: pad the graph mask
            # This happens during generation when we go beyond the original input
            pad_size = key_len - graph_seq_len
            graph_mask_slice = graph_attention_mask[:, :seq_len, :]
            # Pad on the right (key dimension) with False
            graph_mask_padded = torch.nn.functional.pad(graph_mask_slice, (0, pad_size), value=False)
            graph_mask_expanded = graph_mask_padded.unsqueeze(1)

            if graph_mask_mode == 'add':
                final_mask = final_mask & ~graph_mask_expanded
            else:  # 'replace'
                graph_block_mask = ~graph_mask_expanded
                final_mask = final_mask | graph_block_mask
        # If seq_len > graph_seq_len, we're generating beyond the original input
        # so we don't apply graph mask to those new positions

    return final_mask


def create_sliding_window_causal_mask(
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    cache_position: torch.LongTensor,
    config,
    past_key_values=None,
    position_ids=None,
    graph_attention_mask=None,
    or_mask_function=None,
    graph_mask_mode='replace',  # 'add' or 'replace'
):
    window_size = config.sliding_window
    batch_size, seq_len = input_embeds.shape[:2]
    key_len = cache_position[-1] + 1
    query_pos = cache_position.view(seq_len, 1)
    key_pos = torch.arange(key_len, device=input_embeds.device).view(1, key_len)
    causal_mask = query_pos < key_pos
    sliding_mask = key_pos < (query_pos - window_size)
    combined_mask = causal_mask | sliding_mask
    final_mask = combined_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    if attention_mask is not None and attention_mask.any():
        padding_mask = attention_mask[:, None, None, :] == 0 # Invert padding mask
        final_mask = final_mask | padding_mask

    # Apply or_mask_function for special token type handling (e.g., image tokens)
    if or_mask_function is not None:
        or_mask = torch.zeros_like(final_mask, dtype=torch.bool)
        for b in range(batch_size):
            for q in range(seq_len):
                for kv in range(key_len):
                    # Convert indices to tensors for the mask function
                    b_t = torch.tensor(b, device=input_embeds.device)
                    q_t = torch.tensor(q, device=input_embeds.device)
                    kv_t = torch.tensor(kv, device=input_embeds.device)
                    mask_val = or_mask_function(b_t, 0, q_t, kv_t)
                    if mask_val.item() if isinstance(mask_val, torch.Tensor) else mask_val:
                        or_mask[b, :, q, kv] = True
        # Where or_mask is True, we ALLOW attention (set final_mask to False)
        final_mask = final_mask & ~or_mask

    # Apply graph attention mask
    if graph_attention_mask is not None:
        # graph_attention_mask shape: [batch_size, orig_seq_len, orig_seq_len]
        # During generation, key_len may be larger than orig_seq_len
        # We need to handle this by only applying the mask to the overlapping region
        graph_seq_len = graph_attention_mask.shape[1]

        # Only apply graph mask to positions that exist in the graph mask
        if key_len <= graph_seq_len and seq_len <= graph_seq_len:
            # Normal case: mask fits the current sequence
            # Slice the graph mask to match current query and key positions
            graph_mask_slice = graph_attention_mask[:, :seq_len, :key_len]
            graph_mask_expanded = graph_mask_slice.unsqueeze(1)

            # PUNCH HOLE mode: Replace graph region with graph mask, keep causal mask for text
            # Convert graph_attention_mask to mask format (True->False=allow, False->True=block)
            graph_as_final_mask = ~graph_mask_expanded

            # Identify which positions are part of the graph region
            # A position is in graph region if it has any connections (row or column has True values)
            has_graph_connections = (graph_mask_expanded.any(dim=-1, keepdim=True) |
                                   graph_mask_expanded.any(dim=-2, keepdim=True))

            # Replace graph region with graph mask, keep causal mask elsewhere
            final_mask = torch.where(has_graph_connections, graph_as_final_mask, final_mask)

        elif seq_len <= graph_seq_len:
            # Key length exceeds graph mask: pad the graph mask
            # This happens during generation when we go beyond the original input
            pad_size = key_len - graph_seq_len
            graph_mask_slice = graph_attention_mask[:, :seq_len, :]
            # Pad on the right (key dimension) with False
            graph_mask_padded = torch.nn.functional.pad(graph_mask_slice, (0, pad_size), value=False)
            graph_mask_expanded = graph_mask_padded.unsqueeze(1)

            if graph_mask_mode == 'add':
                final_mask = final_mask & ~graph_mask_expanded
            else:  # 'replace'
                graph_block_mask = ~graph_mask_expanded
                final_mask = final_mask | graph_block_mask
        # If seq_len > graph_seq_len, we're generating beyond the original input
        # so we don't apply graph mask to those new positions

    return final_mask


class Gemma3TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [Gemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config=config)
        
        local_config = copy.deepcopy(config)
        local_config.rope_theta = local_config.rope_local_base_freq
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=local_config)
        
    def get_input_embeddings(self):
        return self.embed_tokens
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        graph_attention_mask=None,
        ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
                "graph_attention_mask": graph_attention_mask,
            }

            causal_mask_mapping={
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
        
        hidden_states = inputs_embeds
        
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)
        
        for decoder_layer in self.layers:
            # Select the correct mask for the current layer
            attention_mask_for_layer = causal_mask_mapping[decoder_layer.attention_type]
            
            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings_global,
                position_embeddings_local,
                attention_mask_for_layer, 
                position_ids,
                past_key_values,
                use_cache,
                cache_position,
            )
            
            hidden_states = layer_outputs[0]
            
        hidden_states = self.norm(hidden_states)
        
        return hidden_states


class Gemma3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Gemma3TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        graph_attention_mask=None,
        ):

        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            graph_attention_mask=graph_attention_mask,
        )

        logits = self.lm_head(hidden_states)

        return_data = {
            "logits": logits,
        }

        return return_data
    
    
class Gemma3MultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(config.vision_config.hidden_size, config.text_config.hidden_size)
        )
        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )
        
        self.patches_per_image = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)
        
    def forward(self, vision_outputs):
        
        batch_size, _, seq_length = vision_outputs.shape
        
        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, seq_length, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()
        
        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs) 
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)
        
        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)
        
        projected_vision_outputs = torch.matmul(normed_vision_outputs, self.mm_input_projection_weight)
        return projected_vision_outputs.type_as(vision_outputs)
        
           
    
    




def token_type_ids_mask_function(
    token_type_ids,
    image_group_ids,
    tokens_per_image
):
    if token_type_ids is None:
        return None
    
    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int)-> bool:
        
        safe_idx = torch.where(kv_idx < token_type_ids.shape[1], kv_idx, 0)
        token_type_ids_at_kv_idx = token_type_ids[batch_idx, safe_idx]
        token_type_ids_at_kv_idx = torch.where(kv_idx < token_type_ids.shape[1], token_type_ids_at_kv_idx, 0)
        
        image_group_ids_at_kv_idx = image_group_ids[batch_idx, safe_idx]
        image_group_ids_at_kv_idx = torch.where(kv_idx < image_group_ids.shape[1], image_group_ids_at_kv_idx, -1)
        
        is_image_block = (token_type_ids[batch_idx, q_idx] == 1) & (token_type_ids_at_kv_idx == 1)
        same_image_block = image_group_ids[batch_idx, q_idx] == image_group_ids_at_kv_idx
        
        # bidirectional attention whenever dealing with image tokens
        return is_image_block & same_image_block
    
    return inner_mask
    




class Gemma3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config) # SiglipVisionModel
        self.multi_modal_projector = Gemma3MultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size
        
        self.language_model = Gemma3TextModel(config.text_config) # "Gemma3TextModel"
        
        self.pad_token_id = self.config.text_config.pad_token_id if self.config.text_config.pad_token_id is not None else -1
        
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()
    
    def get_image_features(self, pixel_values):
        vision_outputs = self.vision_tower(pixel_values) # self.vision_tower(pixel_values).last_hidden_states
        image_features = self.multi_modal_projector(vision_outputs)
        return image_features
    
    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features):
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            
        n_image_tokens = special_image_mask.sum()
        
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        
        n_image_features = image_features.shape[0] * image_features.shape[1]
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: {n_image_tokens}, features {n_image_features}"
            )
        
        return special_image_mask
        
    
    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        token_type_ids=None,
        cache_position=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        graph_attention_mask=None,
    ):
        # if (input_ids is None) ^ (inputs_embeds is not None):
        #     raise ValueError("You must specify exactly one of input_ids or inputs_embeds")


        # replace image id with PAD if the image token if OOV, to avoid index-errors
        if input_ids is not None and self.config.image_token_id >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)


        if cache_position is None:
            past_seen_tokens = 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
            
        # merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
            
        # it may already have been prepared by e.g. 'generate'
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # prepare mask arguments
            # graph_mask_mode: 'add' = add graph connections to causal mask (less restrictive)
            #                  'replace' = enforce ONLY graph structure for graph nodes (more restrictive)
            mask_kwargs = {
                "config": self.config.text_config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
                "graph_attention_mask": graph_attention_mask,
                "graph_mask_mode": "replace",  # Use 'replace' mode to enforce graph structure
            }

            if token_type_ids is not None and inputs_embeds.shape[1] != 1:
                
                is_image = (token_type_ids == 1).to(cache_position.device)
                new_image_start = is_image & ~nn.functional.pad(is_image, (1, 0), value=0)[:, :-1]
                image_group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
                image_group_ids = torch.where(
                    is_image, image_group_ids, torch.full_like(token_type_ids, -1, device=is_image.device)
                )
                mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                    token_type_ids.to(cache_position.device), image_group_ids, self.config.mm_tokens_per_image
                )
                
            causal_mask_mapping = {
                "full_attention" : create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs)
            }

        # Store masks for visualization (optional - can be enabled for debugging)
        if hasattr(self, '_save_masks_for_viz') and self._save_masks_for_viz:
            self._stored_masks = {
                'full_attention': causal_mask_mapping['full_attention'].detach().cpu(),
                'sliding_attention': causal_mask_mapping['sliding_attention'].detach().cpu(),
                'graph_attention_mask_input': graph_attention_mask.detach().cpu() if graph_attention_mask is not None else None
            }

        outputs = self.language_model(
            attention_mask=causal_mask_mapping,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache,
            cache_position=cache_position,
            graph_attention_mask=graph_attention_mask,
        )

        return outputs
            
        
        
    

class GraphGemma3ForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Gemma3Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
    
    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        token_type_ids=None,
        cache_position=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        graph_attention_mask=None,
    ):

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            labels=labels,
            cache_position=cache_position,
            graph_attention_mask=graph_attention_mask,
        )

        hidden_states = outputs

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # upcast to float if we need to compute the loss
            logits = logits.float()

            pass


        output_dict = {
            "logits": logits
        }

        return output_dict
    
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        pixel_values=None,
        past_key_values=None,
        cache_position=None,
        inputs_embeds=None,
        labels=None,
        max_new_tokens: int = 100,
        eos_token_id: list = None,
        temperature: float = 1.0,
        top_k: int = 50,
        graph_attention_mask=None,
    ):

        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id


        past_key_values = DynamicCache()

        generated_tokens = input_ids.clone()

        for step in range(max_new_tokens):

            cache_position = torch.arange(
                past_key_values._seen_tokens,
                past_key_values._seen_tokens + input_ids.shape[1],
                device=input_ids.device,
            )

            # 1. Prepare model inputs
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "pixel_values": pixel_values,
                "position_ids": position_ids,
                "inputs_embeds": inputs_embeds,
                "past_key_values": past_key_values,
                "use_cache": True,
                "cache_position": cache_position,
                "graph_attention_mask": graph_attention_mask,
            }
            
            outputs = self(**model_inputs)

            next_token_logits = outputs["logits"][:, -1, :]

            if temperature > 0:
                # Apply temperature scaling
                scaled_logits = next_token_logits / temperature
                
                # Top-K sampling: Keep only the top_k most likely tokens
                top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k)
                
                # Convert logits to probabilities and sample
                probs = F.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1)
                next_token = torch.gather(top_k_indices, -1, next_token_idx)
            else:
                # Greedy search: simply pick the token with the highest logit
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            # 6. Append the new token and prepare for the next iteration
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
            input_ids = next_token # CRITICAL: For the next step, only the new token is the input
            
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype)
                ], dim=1)
            
            # 7. Check for the stopping condition
            if next_token.item() in eos_token_id:
                break
        
        # Concatenate all generated tokens into a single tensor and return
        return generated_tokens

