read_liberty /OpenROAD-flow-scripts/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
read_verilog build/fused-qkv-v3/sky130/ace2_shell_mapped_sta.v
link_design ace2_shell
read_sdc constraints/ace2_rmsnorm_core.sdc

set cache_nets [get_nets qkv_activation_cache_q*]
set act_nets [get_nets proj_act_low_q*]
set cache_cells [get_cells -of_objects $cache_nets]
set act_cells [get_cells -of_objects $act_nets]
puts "ACE2_FUSED_QKV_PATH_COUNTS cache_nets=[llength $cache_nets] act_nets=[llength $act_nets] cache_cells=[llength $cache_cells] act_cells=[llength $act_cells]"
if {[llength $cache_cells] == 0 || [llength $act_cells] == 0} {
    error "missing fused QKV cache timing endpoints"
}
report_checks -from $cache_cells -to $act_cells -path_delay max -digits 4
