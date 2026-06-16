from __future__ import annotations

from feeder_sa_cycles_model import EnergyParams, BWParams, DVFS

def get_energy_params() -> EnergyParams:
    spad_in_read_64b_energy = 35e-12
    spad_out_write_64b_energy = 31e-12
    weight_buf_read_64b_energy = 16e-12
    weight_buf_write_64b_energy = 191e-12
    rram_read_64b_energy = 64.00e-12 # past literature
    rram_write_64b_energy = 5037.5e-12 * 64 # past literature

    return EnergyParams(
        vref = 1.1,
        e_pe_cycle_ref_j=1.833619e-13, # per PE 8b x 8b mult + accum and register
        e_weight_tile_store_bit_ref_j=weight_buf_write_64b_energy/64,
        e_weight_tile_read_bit_ref_j=weight_buf_read_64b_energy/64,
        e_spad_store_bit_ref_j=spad_out_write_64b_energy/64,
        e_spad_read_bit_ref_j=spad_in_read_64b_energy/64,
        e_rram_read_bit_ref_j=rram_read_64b_energy/64,
        e_lane_buf_store_bit_ref_j=1.798701e-13,
        e_lane_buf_read_bit_ref_j=1.798701e-13,
        e_sa_stagger_cycle_ref_j=9.734584e-12,
        e_feeder_ctrl_cycle_ref_j=9.822556e-12,
        e_pg_ctrl_cycle_ref_j=2.579131e-12,
        e_weight_dma_ctrl_64b_ref_j=1.316180e-12,
        p_idle_sa_w = 2.32e-6,
        p_idle_feeder_w = 5.71e-6,
        p_idle_weight_domain_w = 6.452e-6,
        p_idle_pg_w = 1.62e-6,
        p_rram_bank_on_w=20e-6,
        p_rram_bank_off_w=0,
        p_ifmap_bank_on_w=20e-6,
        p_ifmap_bank_off_w=0,
        p_ofmap_bank_on_w=10e-6,
        p_ofmap_bank_off_w=0
    )


def get_bw_params() -> BWParams:
    return BWParams(
        bw_spad_bits_cycle=64,
        bw_rram_bits_cycle=64,
        bw_out_bits_cycle=64,
    )


def get_dvfs_params() -> DVFS:
    # Default DVFS operating point.
    return DVFS(
        freq_sys_hz=500e6,
        volt_sys_v=1.1,
        freq_rram_hz=100e6,
        volt_rram_v=1.1,
        freq_feeder_hz=500e6,
        volt_feeder_v=1.1,
    )
