import logging
from typing import Callable, Dict, Optional, Type

import torch
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.generic_moe import (
    AttentionFactory,
    FMHAImplFactory,
    GenericMoeDecoderLayer,
    GenericMoeLayer,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import RMSNorm, SelectTopk
from rtp_llm.models_py.modules.attention import CausalAttention
from rtp_llm.models_py.modules.embedding import Embedding
from rtp_llm.models_py.modules.factory.fused_moe import FusedMoeFactory
from rtp_llm.models_py.modules.fmha import FMHAImplBase
from rtp_llm.models_py.modules.linear import Linear
from rtp_llm.models_py.modules.mla.mla_attention import MlaAttention
from rtp_llm.models_py.modules.mlp import FusedSiluActDenseMLP
from rtp_llm.models_py.modules.multimodal_embedding import (
    MultimodalDeepstackInjector,
    MultimodalEmbeddingInjector,
)
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W


class Qwen3VLMoeModel(GptModelBase):
    """Qwen3VL MoE model"""

    def __init__(
        self,
        config: GptInitModelParameters,
        weights: ModelWeights,
        attention_type: str = "causal",  # Default attention type
    ):
        super().__init__(config, weights)
        self.attention_type = attention_type
        self.embed_tokens = Embedding(config, weights.get_global_weight(W.embedding))
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        self.multimodal_deepstack_injector = MultimodalDeepstackInjector()
        self.layers = nn.ModuleList(
            [
                GenericMoeDecoderLayer(
                    config, weights.weights[idx], idx, attention_type
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        position_ids = inputs.attention_inputs.combo_position_ids
        token_type_ids = inputs.attention_inputs.combo_tokens_type_ids
        text_tokens_mask = inputs.attention_inputs.text_tokens_mask
        mm_features = inputs.attention_inputs.multimodal_features
        mm_feature_locs = inputs.attention_inputs.mm_features_locs
        mm_deepstack_embeds = inputs.attention_inputs.mm_deepstack_embeds

        inputs_embeds = self.embed_tokens(
            input_ids, position_ids, token_type_ids, text_tokens_mask
        )
        hidden_states = self.multimodal_embedding_injector(
            inputs_embeds, mm_features, mm_feature_locs
        )

        attention_inputs: PyAttentionInputs = inputs.attention_inputs
        impl_method = FMHAImplFactory.get_fmha_impl_method(self.attention_type)
        fmha_impl = getattr(self, impl_method)(attention_inputs)

        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            hidden_states = decoder_layer(
                hidden_states,
                position_ids,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
            hidden_states = self.multimodal_deepstack_injector(
                hidden_states, mm_deepstack_embeds, mm_feature_locs, i
            )

        hidden_states = self.norm(hidden_states)

        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = ["Qwen3VLMoeModel"]
