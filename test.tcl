place_design -unplace
route_design -unroute 
#optimisations
phys_opt_design 
#optimisations
place_design
phys_opt_design
route_design 
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 10 -input_pins -routable_nets -name timing_1 > report_timing_post_route.rpt

