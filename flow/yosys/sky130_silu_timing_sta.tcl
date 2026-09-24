read_liberty /OpenROAD-flow-scripts/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
read_verilog build/silu-timing-3d9615bcdd33/current/ace2_shell_mapped_sta.v
link_design ace2_shell
read_sdc constraints/ace2_rmsnorm_core.sdc
report_checks -path_delay max -digits 4 -group_path_count 5 -endpoint_path_count 5
report_wns
report_tns
