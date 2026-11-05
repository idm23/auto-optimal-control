| Benchmark | Quantity | Indirect (this package) | Reference | Rel. diff | Source | Time (s) |
|---|---|---|---|---|---|---|
| brachistochrone | t_f (g=9.81, x_f=1.0, y_f=1.0) | 0.5828954632 | 0.5828954632 | 0.0e+00 ok | closed form (cycloid) | 0.1 |
| brachistochrone | t_f (g=9.81, x_f=2.0, y_f=0.5) | 0.8566574093 | 0.8566574093 | 9.1e-14 ok | closed form (cycloid) | 0.0 |
| brachistochrone | t_f (g=1.0, x_f=1.0, y_f=2.0) | 2.173540408 | 2.173540408 | 2.3e-14 ok | closed form (cycloid) | 0.0 |
| bryson_denham | J (l=0.2, touch) | 2.24 | 2.24 | 7.9e-16 ok | closed form (Bryson, Denham & Dreyfus 1963) | 0.1 |
| bryson_denham | J (l=0.1, boundary arc) | 4.444444444 | 4.444444444 | 4.0e-16 ok | closed form (Bryson, Denham & Dreyfus 1963) | 2.1 |
| velocity_limit | J (V=1.1) | 11.83111111 | 11.83111111 | 2.0e-14 ok | closed form (order-1 boundary arc) | 0.1 |
| velocity_limit | J (V=1.2) | 7.68 | 7.68 | 6.8e-15 ok | closed form (order-1 boundary arc) | 0.1 |
| velocity_limit | J (V=1.4) | 6.097777778 | 6.097777778 | 4.4e-16 ok | closed form (order-1 boundary arc) | 0.1 |
| velocity_limit | J (V=1.2, |u|<=8.0) | 7.752479139 | 7.752479139 | 3.4e-15 ok | closed form (saturated, then a boundary arc) | 0.1 |
| velocity_limit | J (V=1.25, |u|<=7.0) | 7 | 7 | 1.3e-13 ok | closed form (saturated, then a boundary arc) | 0.1 |
| speed_limit | t_f (V=1.0, D=3.0, u_max=1.0) | 4 | 4 | 3.3e-16 ok | closed form (full power, cruise, brake) | 0.1 |
| speed_limit | t_f (V=2.0, D=10.0, u_max=0.5) | 9 | 9 | 2.0e-16 ok | closed form (full power, cruise, brake) | 0.2 |
| orbit_raising | r(t_f) | 1.525277701 | 1.525277594 | 7.0e-08 ok | YAPSS orbit-raising example (IPOPT); Bryson & Ho 1975 pp. 66-69 give 1.525 | 17.3 |
| earth_mars | m(t_f) [kg] | 603.940156 | 603.935 | 8.5e-06 ok | arXiv:2409.01490 Table 2 (HTS/L2 smoothing, rho = 1e-5) | 9.6 |
| robot_arm | t_f | 9.140911746 | 9.1409 | 1.3e-06 ok | COPS robot arm; GAMS NOA / PROPT (nh = 800 discretization) | 75.3 |
| cart_pole | J (cheapest extremal found) | 57.94478031 | 57.9447803 | 3.8e-11 ok | independent SciPy simulation of Kelly's equations under this u(t) | 62.5 |
| goddard | h(t_f) | 1.012836918 | 1.01283692 | 1.7e-09 ok | Graichen & Petit (2008), collocation with saturation functions | 6.5 |
| goddard | t_f | 0.1988562621 | 0.19885626 | 1.1e-08 ok | Graichen & Petit (2008), collocation with saturation functions | 6.5 |
| goddard | h(t_f), q <= 10 | 1.012717683 | 1.01271727 | 4.1e-07 ok | Graichen & Petit (2008), collocation with saturation functions | 6.0 |
| goddard | t_f, q <= 10 | 0.2040472156 | 0.20407108 | 1.2e-04 ok | Graichen & Petit (2008), collocation with saturation functions | 6.0 |
