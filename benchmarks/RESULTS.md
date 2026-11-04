| Benchmark | Quantity | Indirect (this package) | Reference | Rel. diff | Source | Time (s) |
|---|---|---|---|---|---|---|
| brachistochrone | t_f (g=9.81, x_f=1.0, y_f=1.0) | 0.5828954632 | 0.5828954632 | 0.0e+00 ok | closed form (cycloid) | 15.3 |
| brachistochrone | t_f (g=9.81, x_f=2.0, y_f=0.5) | 0.8566574093 | 0.8566574093 | 9.1e-14 ok | closed form (cycloid) | 15.3 |
| brachistochrone | t_f (g=1.0, x_f=1.0, y_f=2.0) | 2.173540408 | 2.173540408 | 2.3e-14 ok | closed form (cycloid) | 15.3 |
| bryson_denham | J (l=0.2, touch, rho=1e+08) | 2.239926296 | 2.24 | 3.3e-05 ok | closed form (Bryson, Denham & Dreyfus 1963) | 0.5 |
| bryson_denham | J (l=0.2, touch, exact arcs) | 2.24 | 2.24 | 1.4e-15 ok | closed form (Bryson, Denham & Dreyfus 1963) | 0.0 |
| bryson_denham | J (l=0.1, boundary arc, rho=1e+07) | 4.443084757 | 4.444444444 | 3.1e-04 ok | closed form (Bryson, Denham & Dreyfus 1963) | 5.4 |
| bryson_denham | J (l=0.1, boundary arc, exact arcs) | 4.444444444 | 4.444444444 | 6.4e-15 ok | closed form (Bryson, Denham & Dreyfus 1963) | 0.0 |
| velocity_limit | J (V=1.1, exact arcs) | 11.83111111 | 11.83111111 | 1.5e-15 ok | closed form (order-1 boundary arc) | 0.2 |
| velocity_limit | J (V=1.2, exact arcs) | 7.68 | 7.68 | 2.1e-15 ok | closed form (order-1 boundary arc) | 0.2 |
| velocity_limit | J (V=1.4, exact arcs) | 6.097777778 | 6.097777778 | 3.1e-15 ok | closed form (order-1 boundary arc) | 0.2 |
| velocity_limit | J (V=1.2, |u|<=8.0, exact arcs) | 7.752479139 | 7.752479139 | 2.2e-13 ok | closed form (saturated, then a boundary arc) | 0.2 |
| velocity_limit | J (V=1.25, |u|<=7.0, exact arcs) | 7 | 7 | 6.4e-13 ok | closed form (saturated, then a boundary arc) | 0.2 |
| speed_limit | t_f (V=1.0, D=3.0, u_max=1.0, bang-bang exact arcs) | 4 | 4 | 3.3e-16 ok | closed form (full power, cruise, brake) | 1.0 |
| speed_limit | t_f (V=2.0, D=10.0, u_max=0.5, bang-bang exact arcs) | 9 | 9 | 2.0e-16 ok | closed form (full power, cruise, brake) | 0.8 |
| orbit_raising | r(t_f) | 1.525277701 | 1.525277594 | 7.0e-08 ok | YAPSS orbit-raising example (IPOPT); Bryson & Ho 1975 pp. 66-69 give 1.525 | 15.4 |
| earth_mars | m(t_f) [kg] | 603.9401561 | 603.935 | 8.5e-06 ok | arXiv:2409.01490 Table 2 (HTS/L2 smoothing, rho = 1e-5) | 37.8 |
| robot_arm | t_f | 9.140911746 | 9.1409 | 1.3e-06 ok | COPS robot arm; GAMS NOA / PROPT (nh = 800 discretization) | 50.7 |
| cart_pole | J (published extremal) | 58.80766746 | 58.80819 | 8.9e-06 ok | ASSET reproduction of Kelly 2017 (LGL5, 64 segments) | 42.1 |
| cart_pole | J (best of random search) | 57.9447803 | 57.9447803 | 6.4e-12 ok | independent SciPy simulation of Kelly's equations under this u(t) | 31.1 |
| goddard | h(t_f) | 1.012836918 | 1.01283692 | 1.7e-09 ok | Graichen & Petit (2008), collocation with a vanishing penalty | 19.4 |
| goddard | t_f | 0.1988562621 | 0.19885626 | 1.1e-08 ok | Graichen & Petit (2008), collocation with a vanishing penalty | 19.4 |
