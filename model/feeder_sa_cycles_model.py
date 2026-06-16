from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Dict, Any, List


@dataclass(frozen=True)
class FeederCfg:
    ifmap_w: int
    ifmap_h: int
    ifmap_c: int
    ker_size: int
    word_w: int = 8
    stride: int = 1
    padding: int = 0
    num_lanes: int = 8  # SA rows


@dataclass(frozen=True)
class DVFS:
    freq_sys_hz: float
    volt_sys_v: float
    freq_rram_hz: float
    volt_rram_v: float
    freq_feeder_hz: float
    volt_feeder_v: float

@dataclass(frozen=True)
class BWParams:
    # Per-cycle bandwidths at their respective clocks.
    bw_spad_bits_cycle: int = 64
    bw_rram_bits_cycle: int = 64
    bw_out_bits_cycle: int = 64


@dataclass(frozen=True)
class EnergyParams:
    # Calibrated reference operating point
    vref: float
    # Dynamic energies
    e_pe_cycle_ref_j: float
    e_weight_tile_store_bit_ref_j: float
    e_weight_tile_read_bit_ref_j: float
    e_spad_store_bit_ref_j: float
    e_spad_read_bit_ref_j: float
    e_rram_read_bit_ref_j: float
    e_lane_buf_store_bit_ref_j: float
    e_lane_buf_read_bit_ref_j: float
    e_sa_stagger_cycle_ref_j: float
    e_feeder_ctrl_cycle_ref_j: float
    e_pg_ctrl_cycle_ref_j: float
    e_weight_dma_ctrl_64b_ref_j: float
    # Leakage powers
    p_idle_sa_w: float
    p_idle_feeder_w: float
    p_idle_weight_domain_w: float
    p_idle_pg_w: float
    p_rram_bank_on_w: float
    p_rram_bank_off_w: float
    p_ifmap_bank_on_w: float
    p_ifmap_bank_off_w: float
    p_ofmap_bank_on_w: float
    p_ofmap_bank_off_w: float


@dataclass(frozen=True)
class MemoryBankCfg:
    # defaulted for squeezenet
    # ifmap 32KB per bank
    # ofmap 64KB per bank 
    # rram 8KB per bank
    ifmap_total_banks: int = 25
    ofmap_total_banks: int = 6
    rram_total_banks: int = 177
    ifmap_bank_bytes: int = 32 * 1024
    ofmap_bank_bytes: int = 64 * 1024


@dataclass(frozen=True)
class LayerRunCfg:
    feeder: FeederCfg
    channel_out: int
    act_bits: int = 8
    weight_bits: int = 8
    out_bits: int = 32
    C: int = 8
    overlap_weight_fetch: bool = True
    overlap_out: bool = True
    # Optional K-dimension chunking controls for large-K layers.
    # If both are None, model uses full-K behavior (backward-compatible).
    k_chunk_size: int | None = None
    wbuf_capacity_bits: int | None = None
    verbose: bool = False


LEAKAGE_MODE_NO_PG = "no_pg"
LEAKAGE_MODE_LAYER_PG = "layer_pg"
LEAKAGE_MODE_RRAM_PG = "rram_pg"


def _resolve_leakage_mode(leakage_mode: str, per_layer_pg: bool | None) -> str:
    if per_layer_pg is not None:
        return LEAKAGE_MODE_LAYER_PG if per_layer_pg else LEAKAGE_MODE_NO_PG
    if leakage_mode not in (LEAKAGE_MODE_NO_PG, LEAKAGE_MODE_LAYER_PG, LEAKAGE_MODE_RRAM_PG):
        raise ValueError(
            f"Unsupported leakage_mode='{leakage_mode}'. "
            f"Choose from: {LEAKAGE_MODE_NO_PG}, {LEAKAGE_MODE_LAYER_PG}, {LEAKAGE_MODE_RRAM_PG}."
        )
    return leakage_mode


def dynamic_v_scale(v: float, vref: float) -> float:
    # FIXME Replace this rule for real DVFS scaling.
    return (v / vref) ** 2


def static_v_scale(v: float, vref: float) -> float:
    # FIXME Replace this rule for real DVFS scaling.
    return v / vref


def popcount(x: int) -> int:
    if hasattr(int, "bit_count"):
        return int(x).bit_count()
    return bin(int(x)).count("1")


def sa_cycles_fill_drain(K: int, v: int, C: int) -> int:
    """
    Common SA cycle equation (fill + compute + drain), adapted for v valid rows:
      cycles(v) = K + (v-1) + (C-1)   if v>0 else 0
    """
    if v <= 0:
        return 0
    return K + (v - 1) + (C - 1)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _round_up_active_units(required_units: float, total_units: int, granularity: int) -> float:
    if granularity <= 0:
        raise ValueError("granularity must be > 0")
    if required_units <= 0:
        return 0.0
    if granularity == 1:
        return float(min(required_units, total_units))
    grouped_units = math.ceil(float(required_units) / float(granularity)) * granularity
    return float(min(grouped_units, total_units))


def max_divisor_leq(n: int, limit: int) -> int:
    best = 1
    for d in range(1, min(n, limit) + 1):
        if n % d == 0:
            best = d
    return best

def model_layer(
    feeder: FeederCfg,
    channel_out: int,
    dvfs: DVFS,
    bw: BWParams,
    e: EnergyParams,
    act_bits: int = 8,
    weight_bits: int = 8,
    out_bits: int = 32,
    C: int = 8,  # SA cols (output-channel lanes)
    overlap_weight_fetch: bool = True,
    k_chunk_size: int | None = None,
    wbuf_capacity_bits: int | None = None,
    verbose: bool = False,
    per_layer_pg: bool | None = None,
    leakage_mode: str = LEAKAGE_MODE_LAYER_PG,
    mem_banks: MemoryBankCfg | None = None,
    sram_gating_granularity: int = 1,
    rram_gating_granularity: int = 1,
) -> Dict[str, Any]:
    leakage_mode = _resolve_leakage_mode(leakage_mode, per_layer_pg)
    if mem_banks is None:
        mem_banks = MemoryBankCfg()

    ker = feeder.ker_size
    Cin = feeder.ifmap_c
    K = Cin * ker * ker

    out_w = (feeder.ifmap_w + 2 * feeder.padding - ker) // feeder.stride + 1
    out_h = (feeder.ifmap_h + 2 * feeder.padding - ker) // feeder.stride + 1
    out_elems = out_w * out_h * channel_out

    nNtiles = (channel_out + (C - 1)) // C
    # Direct (trace-free) counts for speed.
    out_w_pad = ceil_div(out_w, feeder.num_lanes) * feeder.num_lanes
    spatial_tiles = (out_w_pad // feeder.num_lanes) * out_h

    # Approximate base positions.
    base_positions = spatial_tiles * feeder.ifmap_c * ker

    # Closed-form valid lanes sum (per base position).
    full_tiles = out_w // feeder.num_lanes
    rem = out_w % feeder.num_lanes
    # valid_lanes_per_row = full_tiles * feeder.num_lanes + rem
    # valid_lanes_sum = valid_lanes_per_row * out_h * feeder.ifmap_c * ker

    # Average words per base position using alignment offsets.
    cfg_xlim = ker + (feeder.num_lanes - 1) * feeder.stride
    word_w = feeder.word_w
    words_per_base_avg = sum(ceil_div(cfg_xlim + a, word_w) for a in range(word_w)) / word_w
    word_fetches = int(round(words_per_base_avg * base_positions))

    # Approximate x_loop_ends (used only for reporting).
    x_loop_ends = base_positions
    counts = {
        "word_fetches": word_fetches,
        "x_loop_ends": x_loop_ends,
        "spatial_tiles": spatial_tiles,
        "til_x_count": out_w_pad // feeder.num_lanes,
        "til_y_count": out_h,
        # "valid_lanes_sum": valid_lanes_sum,
    }
    # print(f"DEBUG: word_fetches={word_fetches}, x_loop_ends={x_loop_ends}, spatial_tiles={spatial_tiles}")
    # word fetches means cycles total in the feeder to fetch all the activation data for the layer, and this is determined by the number of base positions (spatial_tiles * ifmap_c * ker) and the average number of words fetched per base position (which depends on the alignment of the base position with respect to the word width). The x_loop_ends is just a count of how many times we end an x loop in the original trace, which corresponds to how many times we need to switch to a new set of activations in the SA. The spatial_tiles is how many output tiles we have in total, which also corresponds to how many times we need to fill/drain the SA for each tile.
    feeder_cycles_inner_total = word_fetches + counts["spatial_tiles"]  # cycles to fetch activation data + swtiching overhead
    # Activation fetch traffic (cycle timing not modeled here).
    bits_per_word = feeder.word_w * act_bits
    ifmap_bits_fetched = counts["word_fetches"] * bits_per_word
    ifmap_tile_bits = K * act_bits

    # Resolve effective K chunking policy.
    # Backward-compatible default: full-K in a single tile.
    if k_chunk_size is not None and k_chunk_size <= 0:
        raise ValueError("k_chunk_size must be > 0 when provided")
    if wbuf_capacity_bits is not None and wbuf_capacity_bits <= 0:
        raise ValueError("wbuf_capacity_bits must be > 0 when provided")
    if k_chunk_size is not None and wbuf_capacity_bits is not None:
        raise ValueError("Provide only one of k_chunk_size or wbuf_capacity_bits")

    if k_chunk_size is not None:
        k_step = min(K, k_chunk_size)
    elif wbuf_capacity_bits is not None:
        k_from_wbuf = wbuf_capacity_bits // (C * weight_bits)
        if k_from_wbuf <= 0:
            raise ValueError(
                "wbuf_capacity_bits is too small for one C-lane weight slice. "
                "Increase capacity or reduce C/weight_bits."
            )
        k_step = min(K, k_from_wbuf)
    else:
        k_step = K

    k_chunks: List[int] = []
    k_left = K
    while k_left > 0:
        k_eff = min(k_step, k_left)
        k_chunks.append(k_eff)
        k_left -= k_eff

    # Use first chunk as representative depth for reporting fields that historically
    # exposed a single i_fill_depth value.
    i_fill_depth = max_divisor_leq(k_chunks[0], 72)  # <=72, divides first chunk
    fill_rows = i_fill_depth // ker
    if fill_rows <= 0:
        fill_rows = 1
    tiles_filled = ceil_div(counts["x_loop_ends"], fill_rows)

    # Compute cycles in K-chunk space; collapses to legacy behavior when one chunk.
    cycles_compute_tile = 0
    weight_bits_per_tile = 0
    weight_cycles_rram = 0
    weight_dma_txn_64b_per_ntile = 0
    for k_eff in k_chunks:
        k_eff_fill = max_divisor_leq(k_eff, 72)
        ifmap_switch_overhead_eff = (k_eff // k_eff_fill) * 2
        cycles_compute_tile += (
            sa_cycles_fill_drain(K=k_eff, v=C, C=C) + C + ifmap_switch_overhead_eff + 1
        )
        w_bits = k_eff * C * weight_bits
        weight_bits_per_tile += w_bits
        weight_cycles_rram += ceil_div(w_bits, bw.bw_rram_bits_cycle)
        weight_dma_txn_64b_per_ntile += ceil_div(w_bits, 64)

    cycles_compute_layer = cycles_compute_tile * counts["spatial_tiles"] * nNtiles
    # if verbose:
        # print(f"DEBUG: cycles_compute_tile={cycles_compute_tile}, cycles_compute_layer={cycles_compute_layer}")
    # Weight fetch: one (possibly K-chunked) KxC tile per Ntile, reused across spatial tiles.
    weight_bits_fetched = nNtiles * weight_bits_per_tile


    # Core cycles within each tile are assumed to be fully overlapped with weight fetch; the slowest determines the tile time.
    core_cycles = cycles_compute_tile * counts["spatial_tiles"] # we update the weight tile after spatial tile count of the ifmap

    # Partial-sum accumulation across K chunks: (Note we are using full K for all our experiments)
    # for each extra K chunk after the first, read+write one psum tile.
    k_tiles = len(k_chunks)
    psum_updates_per_ntile = max(0, k_tiles - 1) * counts["spatial_tiles"]

    psum_tile_bits = C * out_bits
    psum_read_bits_per_ntile = psum_updates_per_ntile * psum_tile_bits
    psum_write_bits_per_ntile = psum_updates_per_ntile * psum_tile_bits
    psum_rw_cycles_per_ntile = (
        ceil_div(
            psum_read_bits_per_ntile + psum_write_bits_per_ntile,
            bw.bw_spad_bits_cycle,
        )
        if (psum_read_bits_per_ntile + psum_write_bits_per_ntile) > 0
        else 0
    )

    # Layer-level psum traffic totals.
    psum_read_bits = psum_read_bits_per_ntile * nNtiles
    psum_write_bits = psum_write_bits_per_ntile * nNtiles

    # inner loop slack
    # overlap ifmap feeder and SA cycles 
    feeder_cycles_inner_total_eff = feeder_cycles_inner_total + psum_rw_cycles_per_ntile
    t_core = max(feeder_cycles_inner_total_eff / dvfs.freq_feeder_hz, core_cycles / dvfs.freq_sys_hz)

    # Weight fetch overlaps with core; one fetch per Ntile.
    # Compare in time domain (sys vs rram clocks).
    # t_core = core_cycles / dvfs.freq_sys_hz
    t_weight = weight_cycles_rram / dvfs.freq_rram_hz
    if overlap_weight_fetch:
        t_ntile = max(t_core, t_weight)
    else:
        t_ntile = t_core + t_weight

    cycles_per_ntile = int(round(t_ntile * dvfs.freq_sys_hz))

    cycles_layer = cycles_per_ntile * nNtiles
    # One-time startup to fetch first weight chunk.
    first_chunk_bits = k_chunks[0] * C * weight_bits
    first_chunk_weight_cycles_rram = ceil_div(first_chunk_bits, bw.bw_rram_bits_cycle)
    t_weight_start = first_chunk_weight_cycles_rram / dvfs.freq_rram_hz
    cycles_layer_with_start = cycles_layer + int(round(t_weight_start * dvfs.freq_sys_hz))

    # Output writeback (final outputs)
    out_bits_written = out_elems * out_bits
    # Output bandwidth already reflected in cycles_out; optional extra overlap with core.
    # If overlap_out is False, treat output as serialized.

    if verbose:
        out_w_pad = ceil_div(out_w, feeder.num_lanes) * feeder.num_lanes
        im2col_m_dim = out_w_pad * out_h
        im2col_k_dim = K
        k_tiles = len(k_chunks)
        print("=== Tile Summary ===")
        print("=== LAYER CONFIGURATION ===")
        print(
            f"Input: {feeder.ifmap_w}x{feeder.ifmap_h}x{feeder.ifmap_c} | "
            f"Output: {out_w}x{out_h}x{channel_out}"
        )
        print(
            f"Input: {feeder.ifmap_w}x{feeder.ifmap_h}x{feeder.ifmap_c} | "
            f"Output(pad): {out_w_pad}x{out_h}x{channel_out}"
        )
        print("---------------------------")
        print(f"Matrix A (im2col padded): [{im2col_m_dim} x {im2col_k_dim}]")
        print(f"Matrix B (Weight):        [{im2col_k_dim} x {channel_out}]")
        print("---------------------------")
        print(f"Tiles M (spatial): {counts['til_y_count']} x {counts['til_x_count']} = {counts['spatial_tiles']}")
        print(f"Tiles N (out ch): {nNtiles} (C={C})")
        print(f"Tiles K (chunks): {k_tiles} (k_step={k_step}, i_fill_depth={i_fill_depth})")
        print("K:", K)
        print("nNtiles (output-channel tiles):", nNtiles)
        print("spatial_tiles (til_y * til_x):", counts["spatial_tiles"])
        print("x_loop_ends:", counts["x_loop_ends"])
        print("i_fill_depth:", i_fill_depth)
        print("tiles_filled (activation tiles):", tiles_filled)
        print("cycles_per_ntile:", cycles_per_ntile)
        print("cycles_layer:", cycles_layer)

    # Energy
    s_sys_dyn = dynamic_v_scale(dvfs.volt_sys_v, e.vref)
    s_rram_dyn = dynamic_v_scale(dvfs.volt_rram_v, e.vref)
    s_feeder_dyn = dynamic_v_scale(dvfs.volt_feeder_v, e.vref)
    s_sys_static = static_v_scale(dvfs.volt_sys_v, e.vref)
    s_rram_static = static_v_scale(dvfs.volt_rram_v, e.vref)
    s_feeder_static = static_v_scale(dvfs.volt_feeder_v, e.vref)

    macs_executed = out_h * out_w * channel_out * K # directly excludes the zero padding...
    # macs_executed = C * K * nNtiles * counts["spatial_tiles"] *v_avg  # total MACs for the layer (all output tiles)

    # if verbose:        
    #     print(f"DEBUG: K={K}, C={C}, spatial_tiles={counts['spatial_tiles']}, nNtiles={nNtiles}")
    #     print(f"DEBUG macs_executed={macs_executed}")
    lanes_blocked_per_row = out_w_pad - out_w
    lanes_blocked = lanes_blocked_per_row * out_h
    util_eff = (out_w / out_w_pad) if out_w_pad > 0 else 0.0
    # Placeholder per-bucket dynamic energy (counts are first-order approximations).
    # e_pe_cycle_ref_j is calibrated per active PE cycle (1 MAC per PE per cycle).
    E_dyn_pe = macs_executed * e.e_pe_cycle_ref_j * s_sys_dyn
    E_dyn_rram_store = 0.0  # no RRAM write traffic modeled for inference path
    E_dyn_rram_read = weight_bits_fetched * e.e_rram_read_bit_ref_j * s_rram_dyn
    E_dyn_w_tile_store = weight_bits_fetched * e.e_weight_tile_store_bit_ref_j * s_rram_dyn
    E_dyn_w_tile_read = weight_bits_fetched * e.e_weight_tile_read_bit_ref_j * s_sys_dyn * counts['til_y_count']
    E_dyn_spad_store = (psum_write_bits + out_bits_written) * e.e_spad_store_bit_ref_j * s_sys_dyn # output spad
    # if (verbose): 
    #     print(f"DEBUG:psum bits: {psum_write_bits}")
    #     print(f"DEBUG:outbits bits: {out_bits_written}")
    #     print(f"DEBUG:ifmap fetched bits: {ifmap_bits_fetched}")
    E_dyn_spad_read = (ifmap_bits_fetched + psum_read_bits) * e.e_spad_read_bit_ref_j * s_feeder_dyn
    # Lane buffer counts are placeholders; refine when exact access counts are defined.
    E_dyn_lane_store = ifmap_tile_bits * nNtiles * counts["til_y_count"] * e.e_lane_buf_store_bit_ref_j * s_feeder_dyn
    E_dyn_lane_read = ifmap_tile_bits * nNtiles * counts["til_y_count"] * e.e_lane_buf_read_bit_ref_j * s_feeder_dyn
    # Explicit control overhead buckets from characterized RTL blocks.
    sa_active_cycles = cycles_compute_layer
    feeder_ctrl_cycles = counts["til_y_count"] * nNtiles * K
    weight_dma_txn_64b = nNtiles * weight_dma_txn_64b_per_ntile
    E_dyn_sa_stagger = sa_active_cycles * e.e_sa_stagger_cycle_ref_j * s_sys_dyn
    # print(f"DEBUG: sa_active_cycles={sa_active_cycles}, macs_executed={macs_executed}")
    E_dyn_feeder_ctrl = feeder_ctrl_cycles * e.e_feeder_ctrl_cycle_ref_j * s_feeder_dyn
    if leakage_mode == LEAKAGE_MODE_RRAM_PG:
        # Approximate one ON/OFF transition sweep per bank per layer.
        E_dyn_pg_ctrl = mem_banks.rram_total_banks * 2 * e.e_pg_ctrl_cycle_ref_j * s_sys_dyn
    else:
        E_dyn_pg_ctrl = 0.0
    E_dyn_weight_dma_ctrl = weight_dma_txn_64b * e.e_weight_dma_ctrl_64b_ref_j * s_rram_dyn
    E_dyn_total = (
        E_dyn_pe
        + E_dyn_rram_store
        + E_dyn_rram_read
        + E_dyn_w_tile_store
        + E_dyn_w_tile_read
        + E_dyn_spad_store
        + E_dyn_spad_read
        + E_dyn_lane_store
        + E_dyn_lane_read
        + E_dyn_sa_stagger
        + E_dyn_feeder_ctrl
        + E_dyn_pg_ctrl
        + E_dyn_weight_dma_ctrl
    )

    # Domain split:
    # - sys: SA/control
    # - feeder: ifmap/ofmap SRAM side
    # - rram: weight domain + RRAM banks
    P_idle_sys = e.p_idle_sa_w
    P_idle_weight = e.p_idle_weight_domain_w
    P_idle_feeder = e.p_idle_feeder_w
    if leakage_mode == LEAKAGE_MODE_RRAM_PG:
        P_idle_sys += e.p_idle_pg_w

    t_layer = cycles_layer / dvfs.freq_sys_hz + t_weight_start
    # SRAM leakage policy by mode.
    if leakage_mode == LEAKAGE_MODE_NO_PG:
        ifmap_on = mem_banks.ifmap_total_banks
        ofmap_on = mem_banks.ofmap_total_banks
    else:
        ifmap_bytes = (feeder.ifmap_w * feeder.ifmap_h * feeder.ifmap_c * act_bits + 7) // 8
        ofmap_bytes = (out_w * out_h * channel_out * out_bits + 7) // 8
        ifmap_required = min(
            (ifmap_bytes + mem_banks.ifmap_bank_bytes - 1) // mem_banks.ifmap_bank_bytes,
            mem_banks.ifmap_total_banks,
        )
        ofmap_required = min(
            (ofmap_bytes + mem_banks.ofmap_bank_bytes - 1) // mem_banks.ofmap_bank_bytes,
            mem_banks.ofmap_total_banks,
        )
        ifmap_on = _round_up_active_units(ifmap_required, mem_banks.ifmap_total_banks, sram_gating_granularity)
        ofmap_on = _round_up_active_units(ofmap_required, mem_banks.ofmap_total_banks, sram_gating_granularity)
    ifmap_off = mem_banks.ifmap_total_banks - ifmap_on
    ofmap_off = mem_banks.ofmap_total_banks - ofmap_on
    P_sram_leak = (
        ifmap_on * e.p_ifmap_bank_on_w
        + ifmap_off * e.p_ifmap_bank_off_w
        + ofmap_on * e.p_ofmap_bank_on_w
        + ofmap_off * e.p_ofmap_bank_off_w
    )

    if leakage_mode == LEAKAGE_MODE_RRAM_PG:
        rram_required = 1.5 if weight_bits_fetched > 0 else 0.0
        rram_on = _round_up_active_units(rram_required, mem_banks.rram_total_banks, rram_gating_granularity)
    else:
        rram_on = mem_banks.rram_total_banks
    rram_off = mem_banks.rram_total_banks - rram_on
    # P_rram_leak = (rram_on * e.p_rram_bank_on_w + rram_off * e.p_rram_bank_off_w) * s_rram_static
    P_rram_leak = (rram_on * e.p_rram_bank_on_w ) * s_rram_static

    E_idle = (
        P_idle_sys * s_sys_static
        + P_idle_weight * s_rram_static
        + P_idle_feeder * s_feeder_static
        + P_sram_leak * s_feeder_static
        + P_rram_leak
    ) * t_layer
    E_total = E_dyn_total + E_idle

    return {
        "inputs": {
            "feeder": asdict(feeder),
            "channel_out": channel_out,
            "dvfs": asdict(dvfs),
            "leakage_mode": leakage_mode,
            "mem_banks": asdict(mem_banks),
            "sram_gating_granularity": sram_gating_granularity,
            "rram_gating_granularity": rram_gating_granularity,
        },
        "derived": {
            "out_w": out_w,
            "out_h": out_h,
            "K": K,
            "nNtiles": nNtiles,
            "k_tiles": len(k_chunks),
            "k_step": k_step,
        },
        "bank_usage": {
            "ifmap_on": ifmap_on,
            "ifmap_off": ifmap_off,
            "ofmap_on": ofmap_on,
            "ofmap_off": ofmap_off,
            "rram_on": rram_on,
            "rram_off": rram_off,
        },
        "trace_counts": {
            "word_fetches": counts["word_fetches"],
            "x_loop_ends": counts["x_loop_ends"],
            "spatial_tiles": counts["spatial_tiles"],
            "i_fill_depth": i_fill_depth,
            "tiles_filled": tiles_filled,
        },
        "traffic_bits": {
            "ifmap_bits_fetched": ifmap_bits_fetched,
            "weight_bits_fetched": weight_bits_fetched,
            "out_bits_written": out_bits_written,
            "psum_read_bits": psum_read_bits,
            "psum_write_bits": psum_write_bits,
        },
        "compute": {
            "macs_executed": macs_executed,
            "cycles_compute_ntile": cycles_compute_tile,
            "cycles_compute_layer": cycles_compute_layer,
            "sa_cycle_model": "K + (v-1) + (C-1)",
        },
        "cycles": {
            "cycles_compute_ntile": cycles_compute_tile,
            "cycles_compute_layer": cycles_compute_layer,
            "weight_cycles_rram": weight_cycles_rram,
            "psum_rw_cycles_per_ntile": psum_rw_cycles_per_ntile,
            "t_weight_s": t_weight,
            "cycles_per_ntile": cycles_per_ntile,
            "cycles_layer": cycles_layer,
            "cycles_layer_with_start": cycles_layer_with_start,
            "t_weight_start_s": t_weight_start,
        },
        "times_s": {"t_layer": t_layer},
        "energy_j": {
            "E_dyn_total": E_dyn_total,
            "E_idle": E_idle,
            "E_total": E_total,
            "buckets": {
                "E_dyn_pe": E_dyn_pe,
                "E_dyn_rram_store": E_dyn_rram_store,
                "E_dyn_rram_read": E_dyn_rram_read,
                "E_dyn_w_tile_store": E_dyn_w_tile_store,
                "E_dyn_w_tile_read": E_dyn_w_tile_read,
                "E_dyn_spad_store": E_dyn_spad_store,
                "E_dyn_spad_read": E_dyn_spad_read,
                "E_dyn_lane_store": E_dyn_lane_store,
                "E_dyn_lane_read": E_dyn_lane_read,
                "E_dyn_sa_stagger": E_dyn_sa_stagger,
                "E_dyn_feeder_ctrl": E_dyn_feeder_ctrl,
                "E_dyn_pg_ctrl": E_dyn_pg_ctrl,
                "E_dyn_weight_dma_ctrl": E_dyn_weight_dma_ctrl,
                "P_sram_leak_w": P_sram_leak,
                "P_rram_leak_w": P_rram_leak,
                "P_idle_non_mem_w": (P_idle_sys + P_idle_weight + P_idle_feeder),
            },
        },
        "perf": {
            "util_eff": util_eff,
            "macs_per_j": (macs_executed / E_total) if E_total > 0 else 0.0,
            "lanes_blocked": lanes_blocked,
        },
    }


def model_layer_sequence(
    layers: List[LayerRunCfg],
    dvfs_sequence: List[DVFS],
    bw: BWParams,
    e: EnergyParams,
    leakage_mode: str = LEAKAGE_MODE_LAYER_PG,
    mem_banks: MemoryBankCfg | None = None,
    sram_gating_granularity: int = 1,
    rram_gating_granularity: int = 1,
) -> Dict[str, Any]:
    if len(layers) != len(dvfs_sequence):
        raise ValueError("layers and dvfs_sequence must have the same length")
    if mem_banks is None:
        mem_banks = MemoryBankCfg()

    layer_reports: List[Dict[str, Any]] = []
    total_e = 0.0
    total_t = 0.0
    total_cycles = 0
    total_macs = 0

    for layer_cfg, dvfs in zip(layers, dvfs_sequence):
        rep = model_layer(
            feeder=layer_cfg.feeder,
            channel_out=layer_cfg.channel_out,
            dvfs=dvfs,
            bw=bw,
            e=e,
            act_bits=layer_cfg.act_bits,
            weight_bits=layer_cfg.weight_bits,
            out_bits=layer_cfg.out_bits,
            C=layer_cfg.C,
            overlap_weight_fetch=layer_cfg.overlap_weight_fetch,
            k_chunk_size=layer_cfg.k_chunk_size,
            wbuf_capacity_bits=layer_cfg.wbuf_capacity_bits,
            # overlap_out=layer_cfg.overlap_out,
            verbose=layer_cfg.verbose,
            leakage_mode=leakage_mode,
            mem_banks=mem_banks,
            sram_gating_granularity=sram_gating_granularity,
            rram_gating_granularity=rram_gating_granularity,
        )
        layer_reports.append(rep)
        total_e += rep["energy_j"]["E_total"]
        total_t += rep["times_s"]["t_layer"]
        total_cycles += rep["cycles"]["cycles_layer_with_start"]
        total_macs += rep["compute"]["macs_executed"]

    return {
        "leakage_mode": leakage_mode,
        "mem_banks": asdict(mem_banks),
        "sram_gating_granularity": sram_gating_granularity,
        "rram_gating_granularity": rram_gating_granularity,
        "layers": layer_reports,
        "totals": {
            "E_total_j": total_e,
            "t_total_s": total_t,
            "cycles_total": total_cycles,
            "macs_total": total_macs,
            "macs_per_j": (total_macs / total_e) if total_e > 0 else 0.0,
        },
    }


def model_matmul_simple(
    m: int,
    k: int,
    n: int,
    dvfs: DVFS,
    bw: BWParams,
    e: EnergyParams,
    act_bits: int = 8,
    weight_bits: int = 8,
    out_bits: int = 32,
    C: int = 8,
    overlap_weight_fetch: bool = True,
    leakage_mode: str = LEAKAGE_MODE_LAYER_PG,
    mem_banks: MemoryBankCfg | None = None,
    sram_gating_granularity: int = 1,
    rram_gating_granularity: int = 1,
) -> Dict[str, Any]:
    """
    Simple deterministic GEMM/MatMul model for transformer-like ops (no im2col):
      A: [m x k], B: [k x n], O: [m x n]
    Assumption: each output-channel tile fetches one RHS tile across K, then computes
    over all M rows (weight tile reused over M). This is intentionally simpler than
    the conv/im2col feeder model.
    """
    leakage_mode = _resolve_leakage_mode(leakage_mode, per_layer_pg=None)
    if mem_banks is None:
        mem_banks = MemoryBankCfg()
    if m <= 0 or k <= 0 or n <= 0:
        raise ValueError("m, k, n must be > 0")

    n_tiles = ceil_div(n, C)
    core_cycles_total = 0
    weight_bits_fetched = 0
    t_core_weight = 0.0
    for t in range(n_tiles):
        c_eff = min(C, n - t * C)
        core_cycles_tile = (sa_cycles_fill_drain(K=k, v=c_eff, C=C) + C + 1) * m
        rhs_bits_tile = k * c_eff * weight_bits
        weight_cycles_tile = ceil_div(rhs_bits_tile, bw.bw_rram_bits_cycle)
        core_cycles_total += core_cycles_tile
        weight_bits_fetched += rhs_bits_tile

        t_core = core_cycles_tile / dvfs.freq_sys_hz
        t_w = weight_cycles_tile / dvfs.freq_rram_hz
        t_core_weight += max(t_core, t_w) if overlap_weight_fetch else (t_core + t_w)

    ifmap_bits_fetched = m * k * act_bits
    out_bits_written = m * n * out_bits
    cycles_ifmap = ceil_div(ifmap_bits_fetched, bw.bw_spad_bits_cycle)
    cycles_out = ceil_div(out_bits_written, bw.bw_out_bits_cycle)
    t_layer = t_core_weight + cycles_ifmap / dvfs.freq_feeder_hz + cycles_out / dvfs.freq_sys_hz

    s_sys_dyn = dynamic_v_scale(dvfs.volt_sys_v, e.vref)
    s_rram_dyn = dynamic_v_scale(dvfs.volt_rram_v, e.vref)
    s_feeder_dyn = dynamic_v_scale(dvfs.volt_feeder_v, e.vref)
    s_sys_static = static_v_scale(dvfs.volt_sys_v, e.vref)
    s_rram_static = static_v_scale(dvfs.volt_rram_v, e.vref)
    s_feeder_static = static_v_scale(dvfs.volt_feeder_v, e.vref)

    macs_executed = m * k * n
    E_dyn_pe = macs_executed * e.e_pe_cycle_ref_j * s_sys_dyn
    E_dyn_rram_read = weight_bits_fetched * e.e_rram_read_bit_ref_j * s_rram_dyn
    E_dyn_w_tile_store = weight_bits_fetched * e.e_weight_tile_store_bit_ref_j * s_rram_dyn
    E_dyn_w_tile_read = weight_bits_fetched * e.e_weight_tile_read_bit_ref_j * s_sys_dyn
    E_dyn_spad_store = (ifmap_bits_fetched + out_bits_written) * e.e_spad_store_bit_ref_j * s_sys_dyn
    E_dyn_spad_read = ifmap_bits_fetched * e.e_spad_read_bit_ref_j * s_feeder_dyn
    E_dyn_sa_stagger = core_cycles_total * e.e_sa_stagger_cycle_ref_j * s_sys_dyn
    E_dyn_feeder_ctrl = (m * n_tiles * k) * e.e_feeder_ctrl_cycle_ref_j * s_feeder_dyn
    E_dyn_pg_ctrl = (
        mem_banks.rram_total_banks * 2 * e.e_pg_ctrl_cycle_ref_j * s_sys_dyn
        if leakage_mode == LEAKAGE_MODE_RRAM_PG
        else 0.0
    )
    E_dyn_weight_dma_ctrl = ceil_div(weight_bits_fetched, 64) * e.e_weight_dma_ctrl_64b_ref_j * s_rram_dyn
    E_dyn_total = (
        E_dyn_pe
        + E_dyn_rram_read
        + E_dyn_w_tile_store
        + E_dyn_w_tile_read
        + E_dyn_spad_store
        + E_dyn_spad_read
        + E_dyn_sa_stagger
        + E_dyn_feeder_ctrl
        + E_dyn_pg_ctrl
        + E_dyn_weight_dma_ctrl
    )

    P_idle_sys = e.p_idle_sa_w + (e.p_idle_pg_w if leakage_mode == LEAKAGE_MODE_RRAM_PG else 0.0)
    P_idle_weight = e.p_idle_weight_domain_w
    P_idle_feeder = e.p_idle_feeder_w
    if leakage_mode == LEAKAGE_MODE_NO_PG:
        ifmap_on = mem_banks.ifmap_total_banks
        ofmap_on = mem_banks.ofmap_total_banks
    else:
        ifmap_bytes = (ifmap_bits_fetched + 7) // 8
        ofmap_bytes = (out_bits_written + 7) // 8
        ifmap_required = min(ceil_div(ifmap_bytes, mem_banks.ifmap_bank_bytes), mem_banks.ifmap_total_banks)
        ofmap_required = min(ceil_div(ofmap_bytes, mem_banks.ofmap_bank_bytes), mem_banks.ofmap_total_banks)
        ifmap_on = _round_up_active_units(ifmap_required, mem_banks.ifmap_total_banks, sram_gating_granularity)
        ofmap_on = _round_up_active_units(ofmap_required, mem_banks.ofmap_total_banks, sram_gating_granularity)
    ifmap_off = mem_banks.ifmap_total_banks - ifmap_on
    ofmap_off = mem_banks.ofmap_total_banks - ofmap_on
    P_sram_leak = (
        ifmap_on * e.p_ifmap_bank_on_w
        + ifmap_off * e.p_ifmap_bank_off_w
        + ofmap_on * e.p_ofmap_bank_on_w
        + ofmap_off * e.p_ofmap_bank_off_w
    )
    if leakage_mode == LEAKAGE_MODE_RRAM_PG:
        rram_required = 1 if weight_bits_fetched > 0 else 0
        rram_on = _round_up_active_units(rram_required, mem_banks.rram_total_banks, rram_gating_granularity)
    else:
        rram_on = mem_banks.rram_total_banks
    rram_off = mem_banks.rram_total_banks - rram_on
    P_rram_leak = (rram_on * e.p_rram_bank_on_w + rram_off * e.p_rram_bank_off_w) * s_rram_static
    E_idle = (
        P_idle_sys * s_sys_static
        + P_idle_weight * s_rram_static
        + P_idle_feeder * s_feeder_static
        + P_sram_leak * s_feeder_static
        + P_rram_leak
    ) * t_layer
    E_total = E_dyn_total + E_idle

    return {
        "inputs": {
            "m": m,
            "k": k,
            "n": n,
            "dvfs": asdict(dvfs),
            "leakage_mode": leakage_mode,
            "sram_gating_granularity": sram_gating_granularity,
            "rram_gating_granularity": rram_gating_granularity,
        },
        "derived": {"n_tiles": n_tiles},
        "traffic_bits": {
            "ifmap_bits_fetched": ifmap_bits_fetched,
            "weight_bits_fetched": weight_bits_fetched,
            "out_bits_written": out_bits_written,
        },
        "compute": {
            "macs_executed": macs_executed,
            "cycles_compute_layer": core_cycles_total,
            "sa_cycle_model": "K + (v-1) + (C-1)",
        },
        "times_s": {"t_layer": t_layer},
        "energy_j": {"E_dyn_total": E_dyn_total, "E_idle": E_idle, "E_total": E_total},
        "perf": {"macs_per_j": (macs_executed / E_total) if E_total > 0 else 0.0},
    }
