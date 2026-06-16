from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from feeder_sa_cycles_model import (
    FeederCfg,
    MemoryBankCfg,
    model_layer,
    LEAKAGE_MODE_NO_PG,
    LEAKAGE_MODE_LAYER_PG,
    LEAKAGE_MODE_RRAM_PG,
)
from energy_config import get_energy_params, get_bw_params, get_dvfs_params


@dataclass(frozen=True)
class ConvSpec:
    name: str
    input_shape: Tuple[int, int, int, int]
    output_shape: Tuple[int, int, int, int]
    weight_shape: Tuple[int, int, int, int]
    stride: int
    padding: int
    groups: int
    is_se_fc: bool


def _infer_padding_1d(input_size: int, output_size: int, kernel_size: int, stride: int) -> int:
    for padding in range(kernel_size + stride + 1):
        modeled = (input_size + 2 * padding - kernel_size) // stride + 1
        if modeled == output_size:
            return padding
    raise ValueError(
        "Cannot infer symmetric padding for "
        f"input={input_size}, output={output_size}, kernel={kernel_size}, stride={stride}"
    )


def _infer_conv_padding(
    input_shape: Tuple[int, int, int, int],
    output_shape: Tuple[int, int, int, int],
    weight_shape: Tuple[int, int, int, int],
    stride: int,
) -> int:
    pad_h = _infer_padding_1d(input_shape[2], output_shape[2], weight_shape[2], stride)
    pad_w = _infer_padding_1d(input_shape[3], output_shape[3], weight_shape[3], stride)
    if pad_h != pad_w:
        raise ValueError(
            "Asymmetric padding is not supported by FeederCfg: "
            f"pad_h={pad_h}, pad_w={pad_w}"
        )
    return pad_h


def build_layers(variant: str) -> Dict[str, Dict[str, Any]]:
    # Static MobileNetV3 conv configs at input 1x3x224x224.
    # Includes depthwise and pointwise convs; SE fc1/fc2 are marked with is_se_fc=True.
    if variant == "large":
        layers = [
            ("features.0.0", (1, 3, 224, 224), (1, 16, 112, 112), (16, 3, 3, 3), 2, 1, False),
            ("features.1.block.0.0", (1, 16, 112, 112), (1, 16, 112, 112), (16, 1, 3, 3), 1, 16, False),
            ("features.1.block.1.0", (1, 16, 112, 112), (1, 16, 112, 112), (16, 16, 1, 1), 1, 1, False),
            ("features.2.block.0.0", (1, 16, 112, 112), (1, 64, 112, 112), (64, 16, 1, 1), 1, 1, False),
            ("features.2.block.1.0", (1, 64, 112, 112), (1, 64, 56, 56), (64, 1, 3, 3), 2, 64, False),
            ("features.2.block.2.0", (1, 64, 56, 56), (1, 24, 56, 56), (24, 64, 1, 1), 1, 1, False),
            ("features.3.block.0.0", (1, 24, 56, 56), (1, 72, 56, 56), (72, 24, 1, 1), 1, 1, False),
            ("features.3.block.1.0", (1, 72, 56, 56), (1, 72, 56, 56), (72, 1, 3, 3), 1, 72, False),
            ("features.3.block.2.0", (1, 72, 56, 56), (1, 24, 56, 56), (24, 72, 1, 1), 1, 1, False),
            ("features.4.block.0.0", (1, 24, 56, 56), (1, 72, 56, 56), (72, 24, 1, 1), 1, 1, False),
            ("features.4.block.1.0", (1, 72, 56, 56), (1, 72, 28, 28), (72, 1, 5, 5), 2, 72, False),
            ("features.4.block.2.fc1", (1, 72, 1, 1), (1, 24, 1, 1), (24, 72, 1, 1), 1, 1, True),
            ("features.4.block.2.fc2", (1, 24, 1, 1), (1, 72, 1, 1), (72, 24, 1, 1), 1, 1, True),
            ("features.4.block.3.0", (1, 72, 28, 28), (1, 40, 28, 28), (40, 72, 1, 1), 1, 1, False),
            ("features.5.block.0.0", (1, 40, 28, 28), (1, 120, 28, 28), (120, 40, 1, 1), 1, 1, False),
            ("features.5.block.1.0", (1, 120, 28, 28), (1, 120, 28, 28), (120, 1, 5, 5), 1, 120, False),
            ("features.5.block.2.fc1", (1, 120, 1, 1), (1, 32, 1, 1), (32, 120, 1, 1), 1, 1, True),
            ("features.5.block.2.fc2", (1, 32, 1, 1), (1, 120, 1, 1), (120, 32, 1, 1), 1, 1, True),
            ("features.5.block.3.0", (1, 120, 28, 28), (1, 40, 28, 28), (40, 120, 1, 1), 1, 1, False),
            ("features.6.block.0.0", (1, 40, 28, 28), (1, 120, 28, 28), (120, 40, 1, 1), 1, 1, False),
            ("features.6.block.1.0", (1, 120, 28, 28), (1, 120, 28, 28), (120, 1, 5, 5), 1, 120, False),
            ("features.6.block.2.fc1", (1, 120, 1, 1), (1, 32, 1, 1), (32, 120, 1, 1), 1, 1, True),
            ("features.6.block.2.fc2", (1, 32, 1, 1), (1, 120, 1, 1), (120, 32, 1, 1), 1, 1, True),
            ("features.6.block.3.0", (1, 120, 28, 28), (1, 40, 28, 28), (40, 120, 1, 1), 1, 1, False),
            ("features.7.block.0.0", (1, 40, 28, 28), (1, 240, 28, 28), (240, 40, 1, 1), 1, 1, False),
            ("features.7.block.1.0", (1, 240, 28, 28), (1, 240, 14, 14), (240, 1, 3, 3), 2, 240, False),
            ("features.7.block.2.0", (1, 240, 14, 14), (1, 80, 14, 14), (80, 240, 1, 1), 1, 1, False),
            ("features.8.block.0.0", (1, 80, 14, 14), (1, 200, 14, 14), (200, 80, 1, 1), 1, 1, False),
            ("features.8.block.1.0", (1, 200, 14, 14), (1, 200, 14, 14), (200, 1, 3, 3), 1, 200, False),
            ("features.8.block.2.0", (1, 200, 14, 14), (1, 80, 14, 14), (80, 200, 1, 1), 1, 1, False),
            ("features.9.block.0.0", (1, 80, 14, 14), (1, 184, 14, 14), (184, 80, 1, 1), 1, 1, False),
            ("features.9.block.1.0", (1, 184, 14, 14), (1, 184, 14, 14), (184, 1, 3, 3), 1, 184, False),
            ("features.9.block.2.0", (1, 184, 14, 14), (1, 80, 14, 14), (80, 184, 1, 1), 1, 1, False),
            ("features.10.block.0.0", (1, 80, 14, 14), (1, 184, 14, 14), (184, 80, 1, 1), 1, 1, False),
            ("features.10.block.1.0", (1, 184, 14, 14), (1, 184, 14, 14), (184, 1, 3, 3), 1, 184, False),
            ("features.10.block.2.0", (1, 184, 14, 14), (1, 80, 14, 14), (80, 184, 1, 1), 1, 1, False),
            ("features.11.block.0.0", (1, 80, 14, 14), (1, 480, 14, 14), (480, 80, 1, 1), 1, 1, False),
            ("features.11.block.1.0", (1, 480, 14, 14), (1, 480, 14, 14), (480, 1, 3, 3), 1, 480, False),
            ("features.11.block.2.fc1", (1, 480, 1, 1), (1, 120, 1, 1), (120, 480, 1, 1), 1, 1, True),
            ("features.11.block.2.fc2", (1, 120, 1, 1), (1, 480, 1, 1), (480, 120, 1, 1), 1, 1, True),
            ("features.11.block.3.0", (1, 480, 14, 14), (1, 112, 14, 14), (112, 480, 1, 1), 1, 1, False),
            ("features.12.block.0.0", (1, 112, 14, 14), (1, 672, 14, 14), (672, 112, 1, 1), 1, 1, False),
            ("features.12.block.1.0", (1, 672, 14, 14), (1, 672, 14, 14), (672, 1, 3, 3), 1, 672, False),
            ("features.12.block.2.fc1", (1, 672, 1, 1), (1, 168, 1, 1), (168, 672, 1, 1), 1, 1, True),
            ("features.12.block.2.fc2", (1, 168, 1, 1), (1, 672, 1, 1), (672, 168, 1, 1), 1, 1, True),
            ("features.12.block.3.0", (1, 672, 14, 14), (1, 112, 14, 14), (112, 672, 1, 1), 1, 1, False),
            ("features.13.block.0.0", (1, 112, 14, 14), (1, 672, 14, 14), (672, 112, 1, 1), 1, 1, False),
            ("features.13.block.1.0", (1, 672, 14, 14), (1, 672, 7, 7), (672, 1, 5, 5), 2, 672, False),
            ("features.13.block.2.fc1", (1, 672, 1, 1), (1, 168, 1, 1), (168, 672, 1, 1), 1, 1, True),
            ("features.13.block.2.fc2", (1, 168, 1, 1), (1, 672, 1, 1), (672, 168, 1, 1), 1, 1, True),
            ("features.13.block.3.0", (1, 672, 7, 7), (1, 160, 7, 7), (160, 672, 1, 1), 1, 1, False),
            ("features.14.block.0.0", (1, 160, 7, 7), (1, 960, 7, 7), (960, 160, 1, 1), 1, 1, False),
            ("features.14.block.1.0", (1, 960, 7, 7), (1, 960, 7, 7), (960, 1, 5, 5), 1, 960, False),
            ("features.14.block.2.fc1", (1, 960, 1, 1), (1, 240, 1, 1), (240, 960, 1, 1), 1, 1, True),
            ("features.14.block.2.fc2", (1, 240, 1, 1), (1, 960, 1, 1), (960, 240, 1, 1), 1, 1, True),
            ("features.14.block.3.0", (1, 960, 7, 7), (1, 160, 7, 7), (160, 960, 1, 1), 1, 1, False),
            ("features.15.block.0.0", (1, 160, 7, 7), (1, 960, 7, 7), (960, 160, 1, 1), 1, 1, False),
            ("features.15.block.1.0", (1, 960, 7, 7), (1, 960, 7, 7), (960, 1, 5, 5), 1, 960, False),
            ("features.15.block.2.fc1", (1, 960, 1, 1), (1, 240, 1, 1), (240, 960, 1, 1), 1, 1, True),
            ("features.15.block.2.fc2", (1, 240, 1, 1), (1, 960, 1, 1), (960, 240, 1, 1), 1, 1, True),
            ("features.15.block.3.0", (1, 960, 7, 7), (1, 160, 7, 7), (160, 960, 1, 1), 1, 1, False),
            ("features.16.0", (1, 160, 7, 7), (1, 960, 7, 7), (960, 160, 1, 1), 1, 1, False),
        ]
    elif variant == "small":
        layers = [
            ("features.0.0", (1, 3, 224, 224), (1, 16, 112, 112), (16, 3, 3, 3), 2, 1, False),
            ("features.1.block.0.0", (1, 16, 112, 112), (1, 16, 56, 56), (16, 1, 3, 3), 2, 16, False),
            ("features.1.block.1.fc1", (1, 16, 1, 1), (1, 8, 1, 1), (8, 16, 1, 1), 1, 1, True),
            ("features.1.block.1.fc2", (1, 8, 1, 1), (1, 16, 1, 1), (16, 8, 1, 1), 1, 1, True),
            ("features.1.block.2.0", (1, 16, 56, 56), (1, 16, 56, 56), (16, 16, 1, 1), 1, 1, False),
            ("features.2.block.0.0", (1, 16, 56, 56), (1, 72, 56, 56), (72, 16, 1, 1), 1, 1, False),
            ("features.2.block.1.0", (1, 72, 56, 56), (1, 72, 28, 28), (72, 1, 3, 3), 2, 72, False),
            ("features.2.block.2.0", (1, 72, 28, 28), (1, 24, 28, 28), (24, 72, 1, 1), 1, 1, False),
            ("features.3.block.0.0", (1, 24, 28, 28), (1, 88, 28, 28), (88, 24, 1, 1), 1, 1, False),
            ("features.3.block.1.0", (1, 88, 28, 28), (1, 88, 28, 28), (88, 1, 3, 3), 1, 88, False),
            ("features.3.block.2.0", (1, 88, 28, 28), (1, 24, 28, 28), (24, 88, 1, 1), 1, 1, False),
            ("features.4.block.0.0", (1, 24, 28, 28), (1, 96, 28, 28), (96, 24, 1, 1), 1, 1, False),
            ("features.4.block.1.0", (1, 96, 28, 28), (1, 96, 14, 14), (96, 1, 5, 5), 2, 96, False),
            ("features.4.block.2.fc1", (1, 96, 1, 1), (1, 24, 1, 1), (24, 96, 1, 1), 1, 1, True),
            ("features.4.block.2.fc2", (1, 24, 1, 1), (1, 96, 1, 1), (96, 24, 1, 1), 1, 1, True),
            ("features.4.block.3.0", (1, 96, 14, 14), (1, 40, 14, 14), (40, 96, 1, 1), 1, 1, False),
            ("features.5.block.0.0", (1, 40, 14, 14), (1, 240, 14, 14), (240, 40, 1, 1), 1, 1, False),
            ("features.5.block.1.0", (1, 240, 14, 14), (1, 240, 14, 14), (240, 1, 5, 5), 1, 240, False),
            ("features.5.block.2.fc1", (1, 240, 1, 1), (1, 64, 1, 1), (64, 240, 1, 1), 1, 1, True),
            ("features.5.block.2.fc2", (1, 64, 1, 1), (1, 240, 1, 1), (240, 64, 1, 1), 1, 1, True),
            ("features.5.block.3.0", (1, 240, 14, 14), (1, 40, 14, 14), (40, 240, 1, 1), 1, 1, False),
            ("features.6.block.0.0", (1, 40, 14, 14), (1, 240, 14, 14), (240, 40, 1, 1), 1, 1, False),
            ("features.6.block.1.0", (1, 240, 14, 14), (1, 240, 14, 14), (240, 1, 5, 5), 1, 240, False),
            ("features.6.block.2.fc1", (1, 240, 1, 1), (1, 64, 1, 1), (64, 240, 1, 1), 1, 1, True),
            ("features.6.block.2.fc2", (1, 64, 1, 1), (1, 240, 1, 1), (240, 64, 1, 1), 1, 1, True),
            ("features.6.block.3.0", (1, 240, 14, 14), (1, 40, 14, 14), (40, 240, 1, 1), 1, 1, False),
            ("features.7.block.0.0", (1, 40, 14, 14), (1, 120, 14, 14), (120, 40, 1, 1), 1, 1, False),
            ("features.7.block.1.0", (1, 120, 14, 14), (1, 120, 14, 14), (120, 1, 5, 5), 1, 120, False),
            ("features.7.block.2.fc1", (1, 120, 1, 1), (1, 32, 1, 1), (32, 120, 1, 1), 1, 1, True),
            ("features.7.block.2.fc2", (1, 32, 1, 1), (1, 120, 1, 1), (120, 32, 1, 1), 1, 1, True),
            ("features.7.block.3.0", (1, 120, 14, 14), (1, 48, 14, 14), (48, 120, 1, 1), 1, 1, False),
            ("features.8.block.0.0", (1, 48, 14, 14), (1, 144, 14, 14), (144, 48, 1, 1), 1, 1, False),
            ("features.8.block.1.0", (1, 144, 14, 14), (1, 144, 14, 14), (144, 1, 5, 5), 1, 144, False),
            ("features.8.block.2.fc1", (1, 144, 1, 1), (1, 40, 1, 1), (40, 144, 1, 1), 1, 1, True),
            ("features.8.block.2.fc2", (1, 40, 1, 1), (1, 144, 1, 1), (144, 40, 1, 1), 1, 1, True),
            ("features.8.block.3.0", (1, 144, 14, 14), (1, 48, 14, 14), (48, 144, 1, 1), 1, 1, False),
            ("features.9.block.0.0", (1, 48, 14, 14), (1, 288, 14, 14), (288, 48, 1, 1), 1, 1, False),
            ("features.9.block.1.0", (1, 288, 14, 14), (1, 288, 7, 7), (288, 1, 5, 5), 2, 288, False),
            ("features.9.block.2.fc1", (1, 288, 1, 1), (1, 72, 1, 1), (72, 288, 1, 1), 1, 1, True),
            ("features.9.block.2.fc2", (1, 72, 1, 1), (1, 288, 1, 1), (288, 72, 1, 1), 1, 1, True),
            ("features.9.block.3.0", (1, 288, 7, 7), (1, 96, 7, 7), (96, 288, 1, 1), 1, 1, False),
            ("features.10.block.0.0", (1, 96, 7, 7), (1, 576, 7, 7), (576, 96, 1, 1), 1, 1, False),
            ("features.10.block.1.0", (1, 576, 7, 7), (1, 576, 7, 7), (576, 1, 5, 5), 1, 576, False),
            ("features.10.block.2.fc1", (1, 576, 1, 1), (1, 144, 1, 1), (144, 576, 1, 1), 1, 1, True),
            ("features.10.block.2.fc2", (1, 144, 1, 1), (1, 576, 1, 1), (576, 144, 1, 1), 1, 1, True),
            ("features.10.block.3.0", (1, 576, 7, 7), (1, 96, 7, 7), (96, 576, 1, 1), 1, 1, False),
            ("features.11.block.0.0", (1, 96, 7, 7), (1, 576, 7, 7), (576, 96, 1, 1), 1, 1, False),
            ("features.11.block.1.0", (1, 576, 7, 7), (1, 576, 7, 7), (576, 1, 5, 5), 1, 576, False),
            ("features.11.block.2.fc1", (1, 576, 1, 1), (1, 144, 1, 1), (144, 576, 1, 1), 1, 1, True),
            ("features.11.block.2.fc2", (1, 144, 1, 1), (1, 576, 1, 1), (576, 144, 1, 1), 1, 1, True),
            ("features.11.block.3.0", (1, 576, 7, 7), (1, 96, 7, 7), (96, 576, 1, 1), 1, 1, False),
            ("features.12.0", (1, 96, 7, 7), (1, 576, 7, 7), (576, 96, 1, 1), 1, 1, False),
        ]
    else:
        raise ValueError(f"Unsupported variant='{variant}'. Choose from: large, small")

    spec: Dict[str, Dict[str, Any]] = {}
    for name, inp, out, w, s, g, is_se_fc in layers:
        spec[name] = {
            "type": "Conv2d",
            "input_shape": inp,
            "output_shape": out,
            "weight_shape": w,
            "stride": s,
            "padding": _infer_conv_padding(inp, out, w, s),
            "groups": g,
            "is_se_fc": is_se_fc,
        }
    return spec


def conv_specs_from_config(spec: Dict[str, Dict[str, Any]]) -> List[ConvSpec]:
    out: List[ConvSpec] = []
    for name, cfg in spec.items():
        out.append(
            ConvSpec(
                name=name,
                input_shape=cfg["input_shape"],
                output_shape=cfg["output_shape"],
                weight_shape=cfg["weight_shape"],
                stride=cfg.get("stride", 1),
                padding=cfg.get("padding", 0),
                groups=cfg.get("groups", 1),
                is_se_fc=bool(cfg.get("is_se_fc", False)),
            )
        )
    return out
