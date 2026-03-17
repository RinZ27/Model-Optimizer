# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lightweight fake base model for offline speculative decoding training."""

import json
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import transformers
from safetensors.torch import load_file as safetensors_load_file
from transformers import PretrainedConfig, PreTrainedModel


@dataclass
class FakeBaseArguments:
    """Arguments for FakeBaseModel used during offline speculative decoding training.

    Pass ``--use_fake_base_model`` to enable. Override the default weight key names for models
    that use a non-standard layout (e.g. VLMs with a ``language_model`` prefix).
    """

    use_fake_base_model: bool = field(
        default=False,
        metadata={
            "help": (
                "Use FakeBaseModel for offline training instead of loading full model weights. "
                "Only effective when --offline_data_path is set."
            )
        },
    )
    lm_head_key: str = field(
        default="lm_head.weight",
        metadata={"help": "Safetensors key for the lm_head weight in the checkpoint."},
    )
    embed_tokens_key: str = field(
        default="model.embed_tokens.weight",
        metadata={"help": "Safetensors key for the embed_tokens weight in the checkpoint."},
    )
    index_filename: str = field(
        default="model.safetensors.index.json",
        metadata={"help": "Name of the sharded safetensors index JSON file."},
    )
    base_config_attr: str | None = field(
        default=None,
        metadata={
            "help": (
                "Attribute name on model_config to use as the base config "
                "(e.g. 'text_config', 'language_config'). "
                "If None, model_config itself is used."
            )
        },
    )


class FakeBaseConfig(PretrainedConfig):
    """Minimal config for FakeBaseModel that supports offline speculative decoding training."""

    model_type = "fake_base_model"

    def __init__(
        self,
        num_hidden_layers=None,
        hidden_size=None,
        vocab_size=None,
        max_position_embeddings=None,
        dtype=torch.bfloat16,
        tie_word_embeddings=False,
        **kwargs,
    ):
        """Initialize FakeBaseConfig with minimal model configuration parameters."""
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.dtype = dtype


class FakeBaseModel(PreTrainedModel):
    """Minimal base model for offline speculative decoding.

    Contains only ``lm_head``, ``embed_tokens``, and the minimal config needed by the EAGLE
    training loop. The full model weights are never loaded, keeping memory usage low.

    Weights are loaded from a local HuggingFace checkpoint directory. The weight key names
    default to standard LLaMA-style paths; override ``lm_head_key`` and ``embed_tokens_key``
    for models with a different layout (e.g. VLMs with a ``language_model`` prefix).
    """

    config_class = FakeBaseConfig

    def __init__(self, source: str, args: "FakeBaseArguments"):
        """Load lm_head and embed_tokens from a local HuggingFace checkpoint directory.

        Args:
            source: Path to a local HuggingFace checkpoint directory.
            args: :class:`FakeBaseArguments` controlling key names and config lookup.
        """
        model_config = transformers.AutoConfig.from_pretrained(source)
        base_cfg = (
            getattr(model_config, args.base_config_attr) if args.base_config_attr else model_config
        )
        hf_config = FakeBaseConfig(
            num_hidden_layers=getattr(base_cfg, "num_hidden_layers", None),
            hidden_size=getattr(base_cfg, "hidden_size", None),
            vocab_size=getattr(base_cfg, "vocab_size", None),
            max_position_embeddings=getattr(base_cfg, "max_position_embeddings", None),
            dtype=getattr(base_cfg, "dtype", torch.bfloat16),
            tie_word_embeddings=getattr(base_cfg, "tie_word_embeddings", False),
        )
        super().__init__(hf_config)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        self.model.dtype = hf_config.dtype
        self.embed_tokens = nn.Embedding(hf_config.vocab_size, hf_config.hidden_size)
        self.lm_head = nn.Linear(hf_config.hidden_size, hf_config.vocab_size, bias=False)

        try:
            lm_head_w, embed_tokens_w = self._load_weights(
                source, args.lm_head_key, args.embed_tokens_key, args.index_filename
            )
            assert lm_head_w.shape == (hf_config.vocab_size, hf_config.hidden_size)
            assert embed_tokens_w.shape == (hf_config.vocab_size, hf_config.hidden_size)
            self.lm_head.weight.data.copy_(lm_head_w)
            self.embed_tokens.weight.data.copy_(embed_tokens_w)
        except Exception as e:
            raise ValueError(f"Failed to initialize lm_head and embed_tokens: {e}")

    def _load_weights(
        self,
        source: str,
        lm_head_key: str,
        embed_tokens_key: str,
        index_filename: str,
    ):
        """Load lm_head and embed_tokens weights from a local checkpoint directory."""
        index_path = os.path.join(source, index_filename)

        if os.path.isfile(index_path):
            with open(index_path) as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            lm_head_file = weight_map.get(lm_head_key)
            embed_tokens_file = weight_map.get(embed_tokens_key)
            if not lm_head_file or not embed_tokens_file:
                raise RuntimeError(f"{lm_head_key} or {embed_tokens_key} not found in index!")
            lm_head_state = safetensors_load_file(os.path.join(source, lm_head_file), device="cpu")
            embed_tokens_state = safetensors_load_file(
                os.path.join(source, embed_tokens_file), device="cpu"
            )
        else:
            raise FileNotFoundError(f"No {index_filename} found in {source!r}.")

        return lm_head_state[lm_head_key], embed_tokens_state[embed_tokens_key]

    def forward(self, *args, **kwargs):
        """Not implemented: FakeBaseModel omits full model weights and cannot run inference."""
        raise NotImplementedError("FakeBaseModel forward is not implemented.")
