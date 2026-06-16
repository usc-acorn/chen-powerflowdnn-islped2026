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


def build_layers() -> Dict[str, Dict[str, Any]]:
    # Conv2d+ReLU only; other layers ignored.
    return {
        "features.0": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 3, 224, 224),
            "output_shape": (1, 64, 111, 111),
            "weight_shape": (64, 3, 3, 3),
            "stride": 2,
            "padding": 0,
        },
        "features.3.squeeze": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 64, 55, 55),
            "output_shape": (1, 16, 55, 55),
            "weight_shape": (16, 64, 1, 1),
        },
        "features.3.expand1x1": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 16, 55, 55),
            "output_shape": (1, 64, 55, 55),
            "weight_shape": (64, 16, 1, 1),
            "inputs": ["features.3.squeeze"],
        },
        "features.3.expand3x3": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 16, 55, 55),
            "output_shape": (1, 64, 55, 55),
            "weight_shape": (64, 16, 3, 3),
            "padding": 1,
            "inputs": ["features.3.squeeze"],
        },
        "features.4.squeeze": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 128, 55, 55),
            "output_shape": (1, 16, 55, 55),
            "weight_shape": (16, 128, 1, 1),
        },
        "features.4.expand1x1": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 16, 55, 55),
            "output_shape": (1, 64, 55, 55),
            "weight_shape": (64, 16, 1, 1),
            "inputs": ["features.4.squeeze"],
        },
        "features.4.expand3x3": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 16, 55, 55),
            "output_shape": (1, 64, 55, 55),
            "weight_shape": (64, 16, 3, 3),
            "padding": 1,
            "inputs": ["features.4.squeeze"],
        },
        "features.6.squeeze": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 128, 27, 27),
            "output_shape": (1, 32, 27, 27),
            "weight_shape": (32, 128, 1, 1),
        },
        "features.6.expand1x1": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 32, 27, 27),
            "output_shape": (1, 128, 27, 27),
            "weight_shape": (128, 32, 1, 1),
            "inputs": ["features.6.squeeze"],
        },
        "features.6.expand3x3": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 32, 27, 27),
            "output_shape": (1, 128, 27, 27),
            "weight_shape": (128, 32, 3, 3),
            "padding": 1,
            "inputs": ["features.6.squeeze"],
        },
        "features.7.squeeze": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 256, 27, 27),
            "output_shape": (1, 32, 27, 27),
            "weight_shape": (32, 256, 1, 1),
        },
        "features.7.expand1x1": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 32, 27, 27),
            "output_shape": (1, 128, 27, 27),
            "weight_shape": (128, 32, 1, 1),
            "inputs": ["features.7.squeeze"],
        },
        "features.7.expand3x3": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 32, 27, 27),
            "output_shape": (1, 128, 27, 27),
            "weight_shape": (128, 32, 3, 3),
            "padding": 1,
            "inputs": ["features.7.squeeze"],
        },
        "features.9.squeeze": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 256, 13, 13),
            "output_shape": (1, 48, 13, 13),
            "weight_shape": (48, 256, 1, 1),
        },
        "features.9.expand1x1": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 48, 13, 13),
            "output_shape": (1, 192, 13, 13),
            "weight_shape": (192, 48, 1, 1),
            "inputs": ["features.9.squeeze"],
        },
        "features.9.expand3x3": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 48, 13, 13),
            "output_shape": (1, 192, 13, 13),
            "weight_shape": (192, 48, 3, 3),
            "padding": 1,
            "inputs": ["features.9.squeeze"],
        },
        "features.10.squeeze": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 384, 13, 13),
            "output_shape": (1, 48, 13, 13),
            "weight_shape": (48, 384, 1, 1),
        },
        "features.10.expand1x1": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 48, 13, 13),
            "output_shape": (1, 192, 13, 13),
            "weight_shape": (192, 48, 1, 1),
            "inputs": ["features.10.squeeze"],
        },
        "features.10.expand3x3": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 48, 13, 13),
            "output_shape": (1, 192, 13, 13),
            "weight_shape": (192, 48, 3, 3),
            "padding": 1,
            "inputs": ["features.10.squeeze"],
        },
        "features.11.squeeze": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 384, 13, 13),
            "output_shape": (1, 64, 13, 13),
            "weight_shape": (64, 384, 1, 1),
        },
        "features.11.expand1x1": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 64, 13, 13),
            "output_shape": (1, 256, 13, 13),
            "weight_shape": (256, 64, 1, 1),
            "inputs": ["features.11.squeeze"],
        },
        "features.11.expand3x3": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 64, 13, 13),
            "output_shape": (1, 256, 13, 13),
            "weight_shape": (256, 64, 3, 3),
            "padding": 1,
            "inputs": ["features.11.squeeze"],
        },
        "features.12.squeeze": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 512, 13, 13),
            "output_shape": (1, 64, 13, 13),
            "weight_shape": (64, 512, 1, 1),
        },
        "features.12.expand1x1": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 64, 13, 13),
            "output_shape": (1, 256, 13, 13),
            "weight_shape": (256, 64, 1, 1),
            "inputs": ["features.12.squeeze"],
        },
        "features.12.expand3x3": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 64, 13, 13),
            "output_shape": (1, 256, 13, 13),
            "weight_shape": (256, 64, 3, 3),
            "padding": 1,
            "inputs": ["features.12.squeeze"],
        },
        "classifier.1": {
            "type": "Conv2d+ReLU",
            "input_shape": (1, 512, 13, 13),
            "output_shape": (1, 2, 13, 13),
            "weight_shape": (2, 512, 1, 1),
        },
    }


def conv_layers_from_spec(spec: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    return [(name, cfg) for name, cfg in spec.items() if cfg.get("type") == "Conv2d+ReLU"]

