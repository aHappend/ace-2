read_liberty /OpenROAD-flow-scripts/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
read_verilog build/projection-k-cache-v3/sky130/ace2_shell_mapped_sta.v
link_design ace2_shell
read_sdc constraints/ace2_rmsnorm_core.sdc

set act_cache_cells [get_cells -of_objects [get_nets proj_act_beat_q*]]
set act_select_cells [get_cells -of_objects [get_nets proj_act_low_q*]]
set weight_cache_cells [get_cells -of_objects [get_nets proj_weight_beat_q*]]
set weight_select_cells [get_cells -of_objects [get_nets proj_weight_q*]]

puts "ACE2_CACHE_PATH_COUNTS act_cache=[llength $act_cache_cells] act_select=[llength $act_select_cells] weight_cache=[llength $weight_cache_cells] weight_select=[llength $weight_select_cells]"
puts "ACE2_ACT_CACHE_PATH"
report_checks -from $act_cache_cells -to $act_select_cells -path_delay max -digits 4
puts "ACE2_WEIGHT_CACHE_PATH"
report_checks -from $weight_cache_cells -to $weight_select_cells -path_delay max -digits 4
