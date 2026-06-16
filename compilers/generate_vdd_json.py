from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from math import prod
from typing import Dict, List, Literal, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (
    _PROJECT_ROOT,
    os.path.join(_PROJECT_ROOT, "model"),
    os.path.join(_PROJECT_ROOT, "networks"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from energy_config import get_bw_params, get_dvfs_params, get_energy_params
from feeder_sa_cycles_model import DVFS, FeederCfg, MemoryBankCfg, model_layer, model_matmul_simple
from mobilenet_v3_run import build_layers as build_mobilenet_v3_layers, conv_specs_from_config
from mobilevit_run import build_mobilevit_xxs_ops, _expand_with_attn_matmuls
from resnet18_run import (
    RESNET18_K_CHUNK,
    build_layers as build_resnet18_layers,
    build_resnet18_mem_banks,
    conv_layers_from_spec as resnet_conv_layers,
)
from squeezenet_run import build_layers as build_squeezenet_layers, conv_layers_from_spec as squeezenet_conv_layers
from vf_model import VFModel

IFMAP_BANK_BYTES = 32 * 1024
OFMAP_BANK_BYTES = 64 * 1024
RRAM_BANK_BYTES = 8 * 1024


@dataclass(frozen=True)
class LayerSpec:
    layer_name: str
    channel_out: int
    kind: Literal["conv", "matmul"]
    feeder: FeederCfg | None = None
    k_chunk_size: int | None = None
    m: int | None = None
    k: int | None = None
    n: int | None = None
    repeats: int = 1


def _parse_voltage_list(text: str) -> List[float]:
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _tensor_bytes(shape: Sequence[int], bits: int) -> int:
    return (_numel(shape) * bits + 7) // 8


def _numel(shape: Sequence[int]) -> int:
    return int(prod(int(x) for x in shape))


def _derive_mem_banks_from_tensors(
    tensors: Sequence[Tuple[Sequence[int], Sequence[int], Sequence[int]]],
    act_bits: int = 8,
    out_bits: int = 32,
    weight_bits: int = 8,
) -> MemoryBankCfg:
    max_ifmap_bytes = 0
    max_ofmap_bytes = 0
    max_weight_bytes = 0
    for input_shape, output_shape, weight_shape in tensors:
        max_ifmap_bytes = max(max_ifmap_bytes, _tensor_bytes(input_shape, act_bits))
        max_ofmap_bytes = max(max_ofmap_bytes, _tensor_bytes(output_shape, out_bits))
        max_weight_bytes = max(max_weight_bytes, _tensor_bytes(weight_shape, weight_bits))

    return MemoryBankCfg(
        ifmap_total_banks=max(1, _ceil_div(max_ifmap_bytes, IFMAP_BANK_BYTES)),
        ofmap_total_banks=max(1, _ceil_div(max_ofmap_bytes, OFMAP_BANK_BYTES)),
        rram_total_banks=max(1, _ceil_div(max_weight_bytes, RRAM_BANK_BYTES)),
        ifmap_bank_bytes=IFMAP_BANK_BYTES,
        ofmap_bank_bytes=OFMAP_BANK_BYTES,
    )


def _build_squeezenet_mem_banks() -> MemoryBankCfg:
    return MemoryBankCfg()


def _build_resnet18_mem_banks() -> MemoryBankCfg:
    return build_resnet18_mem_banks(resnet_conv_layers(build_resnet18_layers()))


def _build_mobilenet_v3_small_mem_banks(include_se: bool = False) -> MemoryBankCfg:
    conv_specs = conv_specs_from_config(build_mobilenet_v3_layers("small"))
    if not include_se:
        conv_specs = [s for s in conv_specs if not s.is_se_fc]
    tensors = [(s.input_shape, s.output_shape, s.weight_shape) for s in conv_specs]
    return _derive_mem_banks_from_tensors(tensors)


def _filtered_mobilevit_xxs_ops(
    include_transformer: bool = True,
    include_attn_matmul: bool = True,
    include_head: bool = False,
):
    ops = build_mobilevit_xxs_ops()
    if include_attn_matmul:
        ops = _expand_with_attn_matmuls(ops)
    if not include_transformer:
        ops = [o for o in ops if ((o.op_type not in ("linear", "matmul")) or o.name == "head.fc")]
    if not include_head:
        ops = [o for o in ops if o.name != "head.fc"]
    return ops


def _build_mobilevit_xxs_mem_banks(
    include_transformer: bool = True,
    include_attn_matmul: bool = True,
    include_head: bool = False,
) -> MemoryBankCfg:
    ops = _filtered_mobilevit_xxs_ops(
        include_transformer=include_transformer,
        include_attn_matmul=include_attn_matmul,
        include_head=include_head,
    )
    tensors = [(op.input_shape, op.output_shape, op.weight_shape) for op in ops]
    return _derive_mem_banks_from_tensors(tensors)


def _build_squeezenet_layers() -> List[LayerSpec]:
    layers: List[LayerSpec] = []
    for name, cfg in squeezenet_conv_layers(build_squeezenet_layers()):
        in_shape = cfg["input_shape"]
        w_shape = cfg["weight_shape"]
        stride = cfg.get("stride", 1)
        padding = cfg.get("padding", 0)
        feeder = FeederCfg(
            ifmap_w=in_shape[2],
            ifmap_h=in_shape[3],
            ifmap_c=w_shape[1],
            ker_size=w_shape[2],
            word_w=8,
            stride=stride,
            padding=padding,
            num_lanes=8,
        )
        layers.append(LayerSpec(layer_name=name, channel_out=w_shape[0], kind="conv", feeder=feeder))
    return layers


def _build_resnet18_layers() -> List[LayerSpec]:
    layers: List[LayerSpec] = []
    for name, cfg in resnet_conv_layers(build_resnet18_layers()):
        in_shape = cfg["input_shape"]
        w_shape = cfg["weight_shape"]
        stride = cfg.get("stride", 1)
        padding = cfg.get("padding", 0)
        feeder = FeederCfg(
            ifmap_w=in_shape[2],
            ifmap_h=in_shape[3],
            ifmap_c=w_shape[1],
            ker_size=w_shape[2],
            word_w=8,
            stride=stride,
            padding=padding,
            num_lanes=8,
        )
        layers.append(
            LayerSpec(
                layer_name=name,
                channel_out=w_shape[0],
                kind="conv",
                feeder=feeder,
                k_chunk_size=RESNET18_K_CHUNK,
            )
        )
    return layers


def _build_mobilenet_v3_small_layers(include_se: bool = False, depthwise_1ch: bool = True) -> List[LayerSpec]:
    layers: List[LayerSpec] = []
    conv_specs = conv_specs_from_config(build_mobilenet_v3_layers("small"))
    if not include_se:
        conv_specs = [s for s in conv_specs if not s.is_se_fc]

    for s in conv_specs:
        cin = s.input_shape[1]
        cout = s.output_shape[1]
        is_depthwise = s.groups == cin and cin == cout
        repeats = cin if (depthwise_1ch and is_depthwise) else 1
        run_ifmap_c = 1 if (depthwise_1ch and is_depthwise) else cin
        run_cout = 1 if (depthwise_1ch and is_depthwise) else cout
        feeder = FeederCfg(
            ifmap_w=s.input_shape[3],
            ifmap_h=s.input_shape[2],
            ifmap_c=run_ifmap_c,
            ker_size=s.weight_shape[2],
            word_w=8,
            stride=s.stride,
            padding=s.padding,
            num_lanes=8,
        )
        layers.append(
            LayerSpec(
                layer_name=s.name,
                channel_out=run_cout,
                kind="conv",
                feeder=feeder,
                repeats=repeats,
            )
        )
    return layers


def _build_mobilevit_xxs_layers(
    include_transformer: bool = True,
    include_attn_matmul: bool = True,
    include_head: bool = False,
    depthwise_1ch: bool = True,
) -> List[LayerSpec]:
    ops = _filtered_mobilevit_xxs_ops(
        include_transformer=include_transformer,
        include_attn_matmul=include_attn_matmul,
        include_head=include_head,
    )

    layers: List[LayerSpec] = []
    for op in ops:
        if op.op_type == "conv":
            cin = op.input_shape[1]
            cout = op.output_shape[1]
            is_depthwise = op.groups == cin and cin == cout
            repeats = cin if (depthwise_1ch and is_depthwise) else 1
            run_ifmap_c = 1 if (depthwise_1ch and is_depthwise) else cin
            run_cout = 1 if (depthwise_1ch and is_depthwise) else cout
            feeder = FeederCfg(
                ifmap_w=op.input_shape[3],
                ifmap_h=op.input_shape[2],
                ifmap_c=run_ifmap_c,
                ker_size=op.weight_shape[2],
                word_w=8,
                stride=op.stride,
                padding=op.padding,
                num_lanes=8,
            )
            layers.append(
                LayerSpec(
                    layer_name=op.name,
                    channel_out=run_cout,
                    kind="conv",
                    feeder=feeder,
                    repeats=repeats,
                )
            )
            continue

        if op.op_type == "linear":
            m = int(prod(op.input_shape[:-1])) if len(op.input_shape) > 1 else 1
            k = int(op.input_shape[-1])
            n = int(op.weight_shape[0])
        elif op.op_type == "matmul":
            b, l, d = [int(x) for x in op.input_shape]
            if op.name.endswith("attn.matmul_qk"):
                m, k, n = b * l, d, l
            elif op.name.endswith("attn.matmul_av"):
                m, k, n = b * l, l, d
            else:
                raise ValueError(f"Unsupported MobileViT matmul op: {op.name}")
        else:
            raise ValueError(f"Unsupported MobileViT op type: {op.op_type}")

        layers.append(
            LayerSpec(
                layer_name=op.name,
                channel_out=n,
                kind="matmul",
                m=m,
                k=k,
                n=n,
            )
        )
    return layers


def _mk_dvfs(v_sys: float, v_rram: float, v_feeder: float, vf_sys: VFModel, vf_rram: VFModel, vf_feeder: VFModel) -> DVFS:
    return DVFS(
        freq_sys_hz=vf_sys.f_hz(v_sys),
        volt_sys_v=v_sys,
        freq_rram_hz=vf_rram.f_hz(v_rram),
        volt_rram_v=v_rram,
        freq_feeder_hz=vf_feeder.f_hz(v_feeder),
        volt_feeder_v=v_feeder,
    )


def generate_json(
    model: str,
    v_sys_candidates: List[float],
    v_rram_candidates: List[float],
    v_feeder_candidates: List[float],
    conv_leakage_mode: str,
    matmul_leakage_mode: str,
    sram_gating_granularity: int,
    rram_gating_granularity: int,
) -> Dict[str, object]:
    if model == "squeezenet":
        layers = _build_squeezenet_layers()
        mem_banks = _build_squeezenet_mem_banks()
    elif model == "resnet18":
        layers = _build_resnet18_layers()
        mem_banks = _build_resnet18_mem_banks()
    elif model == "mobilenetv3_small":
        layers = _build_mobilenet_v3_small_layers(include_se=False, depthwise_1ch=True)
        mem_banks = _build_mobilenet_v3_small_mem_banks(include_se=False)
    elif model == "mobilevit_xxs":
        layers = _build_mobilevit_xxs_layers(
            include_transformer=True,
            include_attn_matmul=True,
            include_head=False,
            depthwise_1ch=True,
        )
        mem_banks = _build_mobilevit_xxs_mem_banks(
            include_transformer=True,
            include_attn_matmul=True,
            include_head=False,
        )
    else:
        raise ValueError(f"Unsupported model: {model}")

    e = get_energy_params()
    bw = get_bw_params()
    dvfs_ref = get_dvfs_params()

    vf_sys = VFModel(v_ref=dvfs_ref.volt_sys_v, f_ref_hz=dvfs_ref.freq_sys_hz)
    vf_rram = VFModel(v_ref=dvfs_ref.volt_rram_v, f_ref_hz=dvfs_ref.freq_rram_hz)
    vf_feeder = VFModel(v_ref=dvfs_ref.volt_feeder_v, f_ref_hz=dvfs_ref.freq_feeder_hz)

    out: Dict[str, object] = {
        "meta": {
            "network": model,
            "v_sys_candidates": v_sys_candidates,
            "v_rram_candidates": v_rram_candidates,
            "v_feeder_candidates": v_feeder_candidates,
            "energy_definition": "energy_j is model E_total; leakage_energy_j is model E_idle",
            "conv_leakage_mode": conv_leakage_mode,
            "matmul_leakage_mode": matmul_leakage_mode,
            "sram_gating_granularity": sram_gating_granularity,
            "rram_gating_granularity": rram_gating_granularity,
            "memory_banks": {
                **asdict(mem_banks),
                "rram_bank_bytes": RRAM_BANK_BYTES,
            },
        },
        "layers": {},
    }

    layers_out: Dict[str, object] = out["layers"]  # type: ignore[assignment]
    for layer_idx, layer in enumerate(layers):
        combo_idx = 0
        combos: Dict[str, object] = {}
        for v_sys in v_sys_candidates:
            for v_rram in v_rram_candidates:
                for v_feeder in v_feeder_candidates:
                    dvfs = _mk_dvfs(v_sys, v_rram, v_feeder, vf_sys, vf_rram, vf_feeder)

                    if layer.kind == "conv":
                        if layer.feeder is None:
                            raise ValueError("conv layer missing feeder config")
                        rep = model_layer(
                            feeder=layer.feeder,
                            channel_out=layer.channel_out,
                            dvfs=dvfs,
                            bw=bw,
                            e=e,
                            act_bits=8,
                            weight_bits=8,
                            out_bits=32,
                            C=8,
                            overlap_weight_fetch=True,
                            k_chunk_size=layer.k_chunk_size,
                            verbose=False,
                            leakage_mode=conv_leakage_mode,
                            mem_banks=mem_banks,
                            sram_gating_granularity=sram_gating_granularity,
                            rram_gating_granularity=rram_gating_granularity,
                        )
                    else:
                        if layer.m is None or layer.k is None or layer.n is None:
                            raise ValueError("matmul layer missing m/k/n")
                        rep = model_matmul_simple(
                            m=layer.m,
                            k=layer.k,
                            n=layer.n,
                            dvfs=dvfs,
                            bw=bw,
                            e=e,
                            act_bits=8,
                            weight_bits=8,
                            out_bits=32,
                            C=8,
                            overlap_weight_fetch=True,
                            leakage_mode=matmul_leakage_mode,
                            mem_banks=mem_banks,
                            sram_gating_granularity=sram_gating_granularity,
                            rram_gating_granularity=rram_gating_granularity,
                        )

                    e_total = rep["energy_j"]["E_total"] * layer.repeats
                    t_layer = rep["times_s"]["t_layer"] * layer.repeats
                    leakage_energy = rep["energy_j"]["E_idle"] * layer.repeats
                    leakage_w = leakage_energy / t_layer if t_layer > 0 else 0.0
                    combos[str(combo_idx)] = {
                        "v_sys": v_sys,
                        "v_rram": v_rram,
                        "v_feeder": v_feeder,
                        "energy_j": e_total,
                        "leakage_energy_j": leakage_energy,
                        "Leakage_w": leakage_w,
                        "time_s": t_layer,
                    }
                    combo_idx += 1

        layers_out[str(layer_idx)] = {
            "layer_name": layer.layer_name,
            "channel_out": layer.channel_out,
            "combinations": combos,
        }

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate JSON for all (v_sys, v_rram, v_feeder) combinations for each layer of a selected model."
    )
    parser.add_argument(
        "--model",
        choices=["squeezenet", "resnet18", "mobilenetv3_small", "mobilevit_xxs", "all"],
        default="squeezenet",
        help="Model to sweep, or all to emit one JSON per model.",
    )
    parser.add_argument(
        "--v-sys",
        default="0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3",
        help="Comma-separated v_sys candidates.",
    )
    parser.add_argument(
        "--v-rram",
        default="0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3",
        help="Comma-separated v_rram candidates.",
    )
    parser.add_argument(
        "--v-feeder",
        default="0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3",
        help="Comma-separated v_feeder candidates.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON path (single-model mode). If omitted, uses <model>_vdd_combinations.json in vdd_sweep.",
    )
    parser.add_argument(
        "--conv-leakage-mode",
        choices=["no_pg", "layer_pg", "rram_pg"],
        default="rram_pg",
        help="Leakage/power-gating mode for convolution layers.",
    )
    parser.add_argument(
        "--matmul-leakage-mode",
        choices=["no_pg", "layer_pg", "rram_pg"],
        default="rram_pg",
        help="Leakage/power-gating mode for matmul layers.",
    )
    parser.add_argument(
        "--sram-gating-granularity",
        type=int,
        default=1,
        help="Minimum number of SRAM banks that remain on together when gating is enabled.",
    )
    parser.add_argument(
        "--rram-gating-granularity",
        type=int,
        default=1,
        help="Minimum number of RRAM banks that remain on together when gating is enabled.",
    )
    args = parser.parse_args()

    v_sys = _parse_voltage_list(args.v_sys)
    v_rram = _parse_voltage_list(args.v_rram)
    v_feeder = _parse_voltage_list(args.v_feeder)
    if args.sram_gating_granularity <= 0 or args.rram_gating_granularity <= 0:
        raise ValueError("Gating granularity must be > 0")

    models = (
        ["squeezenet", "resnet18", "mobilenetv3_small", "mobilevit_xxs"]
        if args.model == "all"
        else [args.model]
    )
    for model in models:
        result = generate_json(
            model=model,
            v_sys_candidates=v_sys,
            v_rram_candidates=v_rram,
            v_feeder_candidates=v_feeder,
            conv_leakage_mode=args.conv_leakage_mode,
            matmul_leakage_mode=args.matmul_leakage_mode,
            sram_gating_granularity=args.sram_gating_granularity,
            rram_gating_granularity=args.rram_gating_granularity,
        )
        if args.output and len(models) == 1:
            out_path = args.output
        else:
            out_path = os.path.join(_HERE, f"{model}_vdd_combinations.json")

        dir_name = os.path.dirname(out_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote VDD sweep JSON: {out_path}")

if __name__ == "__main__":
    main()
