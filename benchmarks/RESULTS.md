| Benchmark | Quantity | Indirect (this package) | Reference | Rel. diff | Source | Time (s) |
|---|---|---|---|---|---|---|
| brachistochrone | t_f (g=9.81, x_f=1.0, y_f=1.0) | 0.5828954632 | 0.5828954632 | 0.0e+00 ok | closed form (cycloid) | 0.1 |
| brachistochrone | t_f (g=9.81, x_f=2.0, y_f=0.5) | 0.8566574093 | 0.8566574093 | 9.1e-14 ok | closed form (cycloid) | 0.0 |
| brachistochrone | t_f (g=1.0, x_f=1.0, y_f=2.0) | 2.173540408 | 2.173540408 | 2.3e-14 ok | closed form (cycloid) | 0.0 |
| bryson_denham | J (l=0.2, touch) | 2.24 | 2.24 | 3.8e-15 ok | closed form (Bryson, Denham & Dreyfus 1963) | 0.4 |
| bryson_denham | J (l=0.1, boundary arc) | 4.444444444 | 4.444444444 | 8.0e-16 ok | closed form (Bryson, Denham & Dreyfus 1963) | 1.0 |
| velocity_limit | J (V=1.1) | 11.83111111 | 11.83111111 | 4.8e-15 ok | closed form (order-1 boundary arc) | 0.5 |
| velocity_limit | J (V=1.2) | 7.68 | 7.68 | 2.2e-15 ok | closed form (order-1 boundary arc) | 0.5 |
| velocity_limit | J (V=1.4) | 6.097777778 | 6.097777778 | 4.4e-16 ok | closed form (order-1 boundary arc) | 0.4 |
| velocity_limit | J (V=1.2, |u|<=8.0) | 7.752479139 | 7.752479139 | 1.4e-13 ok | closed form (saturated, then a boundary arc) | 0.6 |
| velocity_limit | J (V=1.25, |u|<=7.0) | 7 | 7 | 1.6e-11 ok | closed form (saturated, then a boundary arc) | 0.6 |
| speed_limit | t_f (V=1.0, D=3.0, u_max=1.0) | 4 | 4 | 4.4e-16 ok | closed form (full power, cruise, brake) | 0.7 |
| speed_limit | t_f (V=2.0, D=10.0, u_max=0.5) | 9 | 9 | 2.0e-16 ok | closed form (full power, cruise, brake) | 0.7 |
| jerk_limited | t_f (J=1.0, A=1.0, V=2.0, D=10.0; a_max, v_max, a_min) | 8 | 8 | 1.1e-16 ok | closed form (S-curve) | 2.1 |
| jerk_limited | t_f (J=1.0, A=1.0, V=2.0, D=4.0; a_max, a_min) | 5.123105626 | 5.123105626 | 0.0e+00 ok | closed form (S-curve) | 1.0 |
| jerk_limited | t_f (J=1.0, A=1.0, V=0.5, D=10.0; v_max) | 21.41421356 | 21.41421356 | 3.3e-16 ok | closed form (S-curve) | 3.4 |
| jerk_limited | t_f (J=1.0, A=1.0, V=2.0, D=1.0; no arcs) | 3.174802104 | 3.174802104 | 6.9e-15 ok | closed form (S-curve) | 0.3 |
| two_link_arm | J (tau <= 20.0/7.0, one after another) | 1840.216858 | 1840.217255 | 2.2e-07 ok | independent SciPy simulation of the arm under this jerk | 0.9 |
| two_link_arm | J (tau <= 17.0/6.5, overlapping) | 7464.658011 | 7464.663629 | 7.5e-07 ok | independent SciPy simulation of the arm under this jerk | 2.7 |
| two_link_arm | J (tau <= 15.0/7.0, nested) | 14418.50076 | 14418.507 | 4.3e-07 ok | independent SciPy simulation of the arm under this jerk | 5.5 |
| orbit_raising | r(t_f) | 1.525277701 | 1.525277594 | 7.0e-08 ok | YAPSS orbit-raising example (IPOPT); Bryson & Ho 1975 pp. 66-69 give 1.525 | 0.6 |
| earth_mars | m(t_f) [kg] | 603.940156 | 603.935 | 8.5e-06 ok | arXiv:2409.01490 Table 2 (HTS/L2 smoothing, rho = 1e-5) | 10.4 |
| robot_arm | t_f | 9.140911746 | 9.1409 | 1.3e-06 ok | COPS robot arm; GAMS NOA / PROPT (nh = 800 discretization) | 44.5 |
| cart_pole | J (cheapest extremal found) | 57.94478031 | 57.9447803 | 3.8e-11 ok | independent SciPy simulation of Kelly's equations under this u(t) | 45.7 |
| goddard | h(t_f) | 1.012836918 | 1.01283692 | 1.7e-09 ok | Graichen & Petit (2008), collocation with saturation functions | 6.8 |
| goddard | t_f | 0.1988562621 | 0.19885626 | 1.1e-08 ok | Graichen & Petit (2008), collocation with saturation functions | 6.8 |
| goddard | h(t_f), q <= 10 | 1.012717683 | 1.01271727 | 4.1e-07 ok | Graichen & Petit (2008), collocation with saturation functions | 4.9 |
| goddard | t_f, q <= 10 | 0.2040472156 | 0.20407108 | 1.2e-04 ok | Graichen & Petit (2008), collocation with saturation functions | 4.9 |
