from __future__ import annotations

from typing import Dict, Any, List, Tuple

from feeder_sa_cycles_model import (
    FeederCfg,
    LayerRunCfg,
    MemoryBankCfg,
    model_layer_sequence,
    LEAKAGE_MODE_NO_PG,
    LEAKAGE_MODE_LAYER_PG,
    LEAKAGE_MODE_RRAM_PG,
)
from energy_config import get_energy_params, get_bw_params, get_dvfs_params

# Weight-buffer K capacity (elements along K per C-lane tile) used for ResNet runs.
# Keep this local to ResNet so existing SqueezeNet scripts stay unchanged.
RESNET18_K_CHUNK = 576
# Default bank sizes used by this project setup.
IFMAP_BANK_BYTES = 32 * 1024
OFMAP_BANK_BYTES = 64 * 1024
RRAM_BANK_BYTES = 8 * 1024


def build_layers() -> Dict[str, Dict[str, Any]]:
    # ResNet-18 conv stack only (skip pooling/add/fc for this model path).
    return {
        "conv1": {
            "type": "Conv2d",
            "input_shape": (1, 3, 224, 224),
            "output_shape": (1, 64, 112, 112),
            "weight_shape": (64, 3, 7, 7),
            "stride": 2,
            "padding": 3,
        },
        "layer1.0.conv1": {
            "type": "Conv2d",
            "input_shape": (1, 64, 56, 56),
            "output_shape": (1, 64, 56, 56),
            "weight_shape": (64, 64, 3, 3),
            "padding": 1,
        },
        "layer1.0.conv2": {
            "type": "Conv2d",
            "input_shape": (1, 64, 56, 56),
            "output_shape": (1, 64, 56, 56),
            "weight_shape": (64, 64, 3, 3),
            "padding": 1,
        },
        "layer1.1.conv1": {
            "type": "Conv2d",
            "input_shape": (1, 64, 56, 56),
            "output_shape": (1, 64, 56, 56),
            "weight_shape": (64, 64, 3, 3),
            "padding": 1,
        },
        "layer1.1.conv2": {
            "type": "Conv2d",
            "input_shape": (1, 64, 56, 56),
            "output_shape": (1, 64, 56, 56),
            "weight_shape": (64, 64, 3, 3),
            "padding": 1,
        },
        "layer2.0.conv1": {
            "type": "Conv2d",
            "input_shape": (1, 64, 56, 56),
            "output_shape": (1, 128, 28, 28),
            "weight_shape": (128, 64, 3, 3),
            "stride": 2,
            "padding": 1,
        },
        "layer2.0.conv2": {
            "type": "Conv2d",
            "input_shape": (1, 128, 28, 28),
            "output_shape": (1, 128, 28, 28),
            "weight_shape": (128, 128, 3, 3),
            "padding": 1,
        },
        "layer2.0.downsample.0": {
            "type": "Conv2d",
            "input_shape": (1, 64, 56, 56),
            "output_shape": (1, 128, 28, 28),
            "weight_shape": (128, 64, 1, 1),
            "stride": 2,
        },
        "layer2.1.conv1": {
            "type": "Conv2d",
            "input_shape": (1, 128, 28, 28),
            "output_shape": (1, 128, 28, 28),
            "weight_shape": (128, 128, 3, 3),
            "padding": 1,
        },
        "layer2.1.conv2": {
            "type": "Conv2d",
            "input_shape": (1, 128, 28, 28),
            "output_shape": (1, 128, 28, 28),
            "weight_shape": (128, 128, 3, 3),
            "padding": 1,
        },
        "layer3.0.conv1": {
            "type": "Conv2d",
            "input_shape": (1, 128, 28, 28),
            "output_shape": (1, 256, 14, 14),
            "weight_shape": (256, 128, 3, 3),
            "stride": 2,
            "padding": 1,
        },
        "layer3.0.conv2": {
            "type": "Conv2d",
            "input_shape": (1, 256, 14, 14),
            "output_shape": (1, 256, 14, 14),
            "weight_shape": (256, 256, 3, 3),
            "padding": 1,
        },
        "layer3.0.downsample.0": {
            "type": "Conv2d",
            "input_shape": (1, 128, 28, 28),
            "output_shape": (1, 256, 14, 14),
            "weight_shape": (256, 128, 1, 1),
            "stride": 2,
        },
        "layer3.1.conv1": {
            "type": "Conv2d",
            "input_shape": (1, 256, 14, 14),
            "output_shape": (1, 256, 14, 14),
            "weight_shape": (256, 256, 3, 3),
            "padding": 1,
        },
        "layer3.1.conv2": {
            "type": "Conv2d",
            "input_shape": (1, 256, 14, 14),
            "output_shape": (1, 256, 14, 14),
            "weight_shape": (256, 256, 3, 3),
            "padding": 1,
        },
        "layer4.0.conv1": {
            "type": "Conv2d",
            "input_shape": (1, 256, 14, 14),
            "output_shape": (1, 512, 7, 7),
            "weight_shape": (512, 256, 3, 3),
            "stride": 2,
            "padding": 1,
        },
        "layer4.0.conv2": {
            "type": "Conv2d",
            "input_shape": (1, 512, 7, 7),
            "output_shape": (1, 512, 7, 7),
            "weight_shape": (512, 512, 3, 3),
            "padding": 1,
        },
        "layer4.0.downsample.0": {
            "type": "Conv2d",
            "input_shape": (1, 256, 14, 14),
            "output_shape": (1, 512, 7, 7),
            "weight_shape": (512, 256, 1, 1),
            "stride": 2,
        },
        "layer4.1.conv1": {
            "type": "Conv2d",
            "input_shape": (1, 512, 7, 7),
            "output_shape": (1, 512, 7, 7),
            "weight_shape": (512, 512, 3, 3),
            "padding": 1,
        },
        "layer4.1.conv2": {
            "type": "Conv2d",
            "input_shape": (1, 512, 7, 7),
            "output_shape": (1, 512, 7, 7),
            "weight_shape": (512, 512, 3, 3),
            "padding": 1,
        },
    }


def conv_layers_from_spec(spec: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    return [(name, cfg) for name, cfg in spec.items() if str(cfg.get("type", "")).startswith("Conv2d")]


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def build_resnet18_mem_banks(
    layers: List[Tuple[str, Dict[str, Any]]],
    act_bits: int = 8,
    out_bits: int = 32,
    weight_bits: int = 8,
) -> MemoryBankCfg:
    max_ifmap_bytes = 0
    max_ofmap_bytes = 0
    max_weight_bytes = 0

    for _name, cfg in layers:
        in_shape = cfg["input_shape"]   # (N, C, H, W)
        out_shape = cfg["output_shape"] # (N, C, H, W)
        w_shape = cfg["weight_shape"]   # (Co, Ci, Kh, Kw)

        ifmap_bytes = (in_shape[1] * in_shape[2] * in_shape[3] * act_bits + 7) // 8
        ofmap_bytes = (out_shape[1] * out_shape[2] * out_shape[3] * out_bits + 7) // 8
        weight_bytes = (w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3] * weight_bits + 7) // 8

        max_ifmap_bytes = max(max_ifmap_bytes, ifmap_bytes)
        max_ofmap_bytes = max(max_ofmap_bytes, ofmap_bytes)
        max_weight_bytes = max(max_weight_bytes, weight_bytes)

    return MemoryBankCfg(
        ifmap_total_banks=max(1, ceil_div(max_ifmap_bytes, IFMAP_BANK_BYTES)),
        ofmap_total_banks=max(1, ceil_div(max_ofmap_bytes, OFMAP_BANK_BYTES)),
        rram_total_banks=max(1, ceil_div(max_weight_bytes, RRAM_BANK_BYTES)),
        ifmap_bank_bytes=IFMAP_BANK_BYTES,
        ofmap_bank_bytes=OFMAP_BANK_BYTES,
    )
