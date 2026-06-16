from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import prod
from typing import Any, Dict, List, Tuple

from feeder_sa_cycles_model import (
    FeederCfg,
    MemoryBankCfg,
    model_matmul_simple,
    model_layer,
    LEAKAGE_MODE_NO_PG,
    LEAKAGE_MODE_LAYER_PG,
    LEAKAGE_MODE_RRAM_PG,
)
from energy_config import get_energy_params, get_bw_params, get_dvfs_params


@dataclass(frozen=True)
class OpSpec:
    op_type: str  # "conv" | "linear" | "matmul"
    name: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    weight_shape: Tuple[int, ...]
    stride: int
    groups: int
    padding: int = 0


def _infer_padding_1d(input_size: int, output_size: int, kernel_size: int, stride: int) -> int:
    for padding in range(kernel_size + stride + 1):
        modeled = (input_size + 2 * padding - kernel_size) // stride + 1
        if modeled == output_size:
            return padding
    raise ValueError(
        "Cannot infer symmetric padding for "
        f"input={input_size}, output={output_size}, kernel={kernel_size}, stride={stride}"
    )


def _infer_conv_padding(op: OpSpec) -> int:
    pad_h = _infer_padding_1d(op.input_shape[2], op.output_shape[2], op.weight_shape[2], op.stride)
    pad_w = _infer_padding_1d(op.input_shape[3], op.output_shape[3], op.weight_shape[3], op.stride)
    if pad_h != pad_w:
        raise ValueError(
            f"Asymmetric padding is not supported by FeederCfg for {op.name}: "
            f"pad_h={pad_h}, pad_w={pad_w}"
        )
    return pad_h


def build_mobilevit_xxs_ops() -> List[OpSpec]:
    # Static MobileViT-XXS op list at input 1x3x224x224.
    # Extracted once via local timm probe and hardcoded for reproducibility.
    ops = [
        ("conv", "stem.conv", (1, 3, 224, 224), (1, 16, 112, 112), (16, 3, 3, 3), 2, 1),
        ("conv", "stages.0.0.conv1_1x1.conv", (1, 16, 112, 112), (1, 32, 112, 112), (32, 16, 1, 1), 1, 1),
        ("conv", "stages.0.0.conv2_kxk.conv", (1, 32, 112, 112), (1, 32, 112, 112), (32, 1, 3, 3), 1, 32),
        ("conv", "stages.0.0.conv3_1x1.conv", (1, 32, 112, 112), (1, 16, 112, 112), (16, 32, 1, 1), 1, 1),
        ("conv", "stages.1.0.conv1_1x1.conv", (1, 16, 112, 112), (1, 32, 112, 112), (32, 16, 1, 1), 1, 1),
        ("conv", "stages.1.0.conv2_kxk.conv", (1, 32, 112, 112), (1, 32, 56, 56), (32, 1, 3, 3), 2, 32),
        ("conv", "stages.1.0.conv3_1x1.conv", (1, 32, 56, 56), (1, 24, 56, 56), (24, 32, 1, 1), 1, 1),
        ("conv", "stages.1.1.conv1_1x1.conv", (1, 24, 56, 56), (1, 48, 56, 56), (48, 24, 1, 1), 1, 1),
        ("conv", "stages.1.1.conv2_kxk.conv", (1, 48, 56, 56), (1, 48, 56, 56), (48, 1, 3, 3), 1, 48),
        ("conv", "stages.1.1.conv3_1x1.conv", (1, 48, 56, 56), (1, 24, 56, 56), (24, 48, 1, 1), 1, 1),
        ("conv", "stages.1.2.conv1_1x1.conv", (1, 24, 56, 56), (1, 48, 56, 56), (48, 24, 1, 1), 1, 1),
        ("conv", "stages.1.2.conv2_kxk.conv", (1, 48, 56, 56), (1, 48, 56, 56), (48, 1, 3, 3), 1, 48),
        ("conv", "stages.1.2.conv3_1x1.conv", (1, 48, 56, 56), (1, 24, 56, 56), (24, 48, 1, 1), 1, 1),
        ("conv", "stages.2.0.conv1_1x1.conv", (1, 24, 56, 56), (1, 48, 56, 56), (48, 24, 1, 1), 1, 1),
        ("conv", "stages.2.0.conv2_kxk.conv", (1, 48, 56, 56), (1, 48, 28, 28), (48, 1, 3, 3), 2, 48),
        ("conv", "stages.2.0.conv3_1x1.conv", (1, 48, 28, 28), (1, 48, 28, 28), (48, 48, 1, 1), 1, 1),
        ("conv", "stages.2.1.conv_kxk.conv", (1, 48, 28, 28), (1, 48, 28, 28), (48, 48, 3, 3), 1, 1),
        ("conv", "stages.2.1.conv_1x1", (1, 48, 28, 28), (1, 64, 28, 28), (64, 48, 1, 1), 1, 1),
        ("linear", "stages.2.1.transformer.0.attn.qkv", (4, 196, 64), (4, 196, 192), (192, 64), 1, 1),
        ("linear", "stages.2.1.transformer.0.attn.proj", (4, 196, 64), (4, 196, 64), (64, 64), 1, 1),
        ("linear", "stages.2.1.transformer.0.mlp.fc1", (4, 196, 64), (4, 196, 128), (128, 64), 1, 1),
        ("linear", "stages.2.1.transformer.0.mlp.fc2", (4, 196, 128), (4, 196, 64), (64, 128), 1, 1),
        ("linear", "stages.2.1.transformer.1.attn.qkv", (4, 196, 64), (4, 196, 192), (192, 64), 1, 1),
        ("linear", "stages.2.1.transformer.1.attn.proj", (4, 196, 64), (4, 196, 64), (64, 64), 1, 1),
        ("linear", "stages.2.1.transformer.1.mlp.fc1", (4, 196, 64), (4, 196, 128), (128, 64), 1, 1),
        ("linear", "stages.2.1.transformer.1.mlp.fc2", (4, 196, 128), (4, 196, 64), (64, 128), 1, 1),
        ("conv", "stages.2.1.conv_proj.conv", (1, 64, 28, 28), (1, 48, 28, 28), (48, 64, 1, 1), 1, 1),
        ("conv", "stages.2.1.conv_fusion.conv", (1, 96, 28, 28), (1, 48, 28, 28), (48, 96, 3, 3), 1, 1),
        ("conv", "stages.3.0.conv1_1x1.conv", (1, 48, 28, 28), (1, 96, 28, 28), (96, 48, 1, 1), 1, 1),
        ("conv", "stages.3.0.conv2_kxk.conv", (1, 96, 28, 28), (1, 96, 14, 14), (96, 1, 3, 3), 2, 96),
        ("conv", "stages.3.0.conv3_1x1.conv", (1, 96, 14, 14), (1, 64, 14, 14), (64, 96, 1, 1), 1, 1),
        ("conv", "stages.3.1.conv_kxk.conv", (1, 64, 14, 14), (1, 64, 14, 14), (64, 64, 3, 3), 1, 1),
        ("conv", "stages.3.1.conv_1x1", (1, 64, 14, 14), (1, 80, 14, 14), (80, 64, 1, 1), 1, 1),
        ("linear", "stages.3.1.transformer.0.attn.qkv", (4, 49, 80), (4, 49, 240), (240, 80), 1, 1),
        ("linear", "stages.3.1.transformer.0.attn.proj", (4, 49, 80), (4, 49, 80), (80, 80), 1, 1),
        ("linear", "stages.3.1.transformer.0.mlp.fc1", (4, 49, 80), (4, 49, 160), (160, 80), 1, 1),
        ("linear", "stages.3.1.transformer.0.mlp.fc2", (4, 49, 160), (4, 49, 80), (80, 160), 1, 1),
        ("linear", "stages.3.1.transformer.1.attn.qkv", (4, 49, 80), (4, 49, 240), (240, 80), 1, 1),
        ("linear", "stages.3.1.transformer.1.attn.proj", (4, 49, 80), (4, 49, 80), (80, 80), 1, 1),
        ("linear", "stages.3.1.transformer.1.mlp.fc1", (4, 49, 80), (4, 49, 160), (160, 80), 1, 1),
        ("linear", "stages.3.1.transformer.1.mlp.fc2", (4, 49, 160), (4, 49, 80), (80, 160), 1, 1),
        ("linear", "stages.3.1.transformer.2.attn.qkv", (4, 49, 80), (4, 49, 240), (240, 80), 1, 1),
        ("linear", "stages.3.1.transformer.2.attn.proj", (4, 49, 80), (4, 49, 80), (80, 80), 1, 1),
        ("linear", "stages.3.1.transformer.2.mlp.fc1", (4, 49, 80), (4, 49, 160), (160, 80), 1, 1),
        ("linear", "stages.3.1.transformer.2.mlp.fc2", (4, 49, 160), (4, 49, 80), (80, 160), 1, 1),
        ("linear", "stages.3.1.transformer.3.attn.qkv", (4, 49, 80), (4, 49, 240), (240, 80), 1, 1),
        ("linear", "stages.3.1.transformer.3.attn.proj", (4, 49, 80), (4, 49, 80), (80, 80), 1, 1),
        ("linear", "stages.3.1.transformer.3.mlp.fc1", (4, 49, 80), (4, 49, 160), (160, 80), 1, 1),
        ("linear", "stages.3.1.transformer.3.mlp.fc2", (4, 49, 160), (4, 49, 80), (80, 160), 1, 1),
        ("conv", "stages.3.1.conv_proj.conv", (1, 80, 14, 14), (1, 64, 14, 14), (64, 80, 1, 1), 1, 1),
        ("conv", "stages.3.1.conv_fusion.conv", (1, 128, 14, 14), (1, 64, 14, 14), (64, 128, 3, 3), 1, 1),
        ("conv", "stages.4.0.conv1_1x1.conv", (1, 64, 14, 14), (1, 128, 14, 14), (128, 64, 1, 1), 1, 1),
        ("conv", "stages.4.0.conv2_kxk.conv", (1, 128, 14, 14), (1, 128, 7, 7), (128, 1, 3, 3), 2, 128),
        ("conv", "stages.4.0.conv3_1x1.conv", (1, 128, 7, 7), (1, 80, 7, 7), (80, 128, 1, 1), 1, 1),
        ("conv", "stages.4.1.conv_kxk.conv", (1, 80, 7, 7), (1, 80, 7, 7), (80, 80, 3, 3), 1, 1),
        ("conv", "stages.4.1.conv_1x1", (1, 80, 7, 7), (1, 96, 7, 7), (96, 80, 1, 1), 1, 1),
        ("linear", "stages.4.1.transformer.0.attn.qkv", (4, 16, 96), (4, 16, 288), (288, 96), 1, 1),
        ("linear", "stages.4.1.transformer.0.attn.proj", (4, 16, 96), (4, 16, 96), (96, 96), 1, 1),
        ("linear", "stages.4.1.transformer.0.mlp.fc1", (4, 16, 96), (4, 16, 192), (192, 96), 1, 1),
        ("linear", "stages.4.1.transformer.0.mlp.fc2", (4, 16, 192), (4, 16, 96), (96, 192), 1, 1),
        ("linear", "stages.4.1.transformer.1.attn.qkv", (4, 16, 96), (4, 16, 288), (288, 96), 1, 1),
        ("linear", "stages.4.1.transformer.1.attn.proj", (4, 16, 96), (4, 16, 96), (96, 96), 1, 1),
        ("linear", "stages.4.1.transformer.1.mlp.fc1", (4, 16, 96), (4, 16, 192), (192, 96), 1, 1),
        ("linear", "stages.4.1.transformer.1.mlp.fc2", (4, 16, 192), (4, 16, 96), (96, 192), 1, 1),
        ("linear", "stages.4.1.transformer.2.attn.qkv", (4, 16, 96), (4, 16, 288), (288, 96), 1, 1),
        ("linear", "stages.4.1.transformer.2.attn.proj", (4, 16, 96), (4, 16, 96), (96, 96), 1, 1),
        ("linear", "stages.4.1.transformer.2.mlp.fc1", (4, 16, 96), (4, 16, 192), (192, 96), 1, 1),
        ("linear", "stages.4.1.transformer.2.mlp.fc2", (4, 16, 192), (4, 16, 96), (96, 192), 1, 1),
        ("conv", "stages.4.1.conv_proj.conv", (1, 96, 7, 7), (1, 80, 7, 7), (80, 96, 1, 1), 1, 1),
        ("conv", "stages.4.1.conv_fusion.conv", (1, 160, 7, 7), (1, 80, 7, 7), (80, 160, 3, 3), 1, 1),
        ("conv", "final_conv.conv", (1, 80, 7, 7), (1, 320, 7, 7), (320, 80, 1, 1), 1, 1),
        ("linear", "head.fc", (1, 320), (1, 1000), (1000, 320), 1, 1),
    ]
    out: List[OpSpec] = []
    for row in ops:
        op = OpSpec(*row)
        if op.op_type == "conv":
            op = OpSpec(
                op_type=op.op_type,
                name=op.name,
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                weight_shape=op.weight_shape,
                stride=op.stride,
                groups=op.groups,
                padding=_infer_conv_padding(op),
            )
        out.append(op)
    return out


def _expand_with_attn_matmuls(ops: List[OpSpec]) -> List[OpSpec]:
    out: List[OpSpec] = []
    for op in ops:
        out.append(op)
        if op.op_type == "linear" and op.name.endswith("attn.qkv"):
            # Input shape is (B, L, D). Add explicit attention matmuls:
            # QK^T -> (B, L, L), then AV -> (B, L, D).
            b, l, d = op.input_shape
            out.append(
                OpSpec(
                    op_type="matmul",
                    name=op.name.replace("attn.qkv", "attn.matmul_qk"),
                    input_shape=(b, l, d),
                    output_shape=(b, l, l),
                    weight_shape=(d, l),
                    stride=1,
                    groups=1,
                )
            )
            out.append(
                OpSpec(
                    op_type="matmul",
                    name=op.name.replace("attn.qkv", "attn.matmul_av"),
                    input_shape=(b, l, d),
                    output_shape=(b, l, d),
                    weight_shape=(l, d),
                    stride=1,
                    groups=1,
                )
            )
    return out
