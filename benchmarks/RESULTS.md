| Benchmark | Quantity | Indirect (this package) | Reference | Rel. diff | Source | Time (s) |
|---|---|---|---|---|---|---|
| brachistochrone | t_f (g=9.81, x_f=1.0, y_f=1.0) | 0.5828954632 | 0.5828954632 | 0.0e+00 ok | closed form (cycloid) | 0.3 |
| brachistochrone | t_f (g=9.81, x_f=2.0, y_f=0.5) | 0.8566574093 | 0.8566574093 | 9.1e-14 ok | closed form (cycloid) | 0.4 |
| brachistochrone | t_f (g=1.0, x_f=1.0, y_f=2.0) | 2.173540408 | 2.173540408 | 2.3e-14 ok | closed form (cycloid) | 0.4 |
| bryson_denham | J (l=0.2, touch, rho=1e+08) | 2.239926296 | 2.24 | 3.3e-05 ok | closed form (Bryson, Denham & Dreyfus 1963) | 0.5 |
| bryson_denham | J (l=0.1, boundary arc, rho=1e+07) | 4.443084757 | 4.444444444 | 3.1e-04 ok | closed form (Bryson, Denham & Dreyfus 1963) | 5.5 |
| orbit_raising | r(t_f) | 1.525277701 | 1.525277594 | 7.0e-08 ok | YAPSS orbit-raising example (IPOPT); Bryson & Ho 1975 pp. 66-69 give 1.525 | 0.4 |
| earth_mars | m(t_f) [kg] | 603.940156 | 603.935 | 8.5e-06 ok | arXiv:2409.01490 Table 2 (HTS/L2 smoothing, rho = 1e-5) | 5.3 |
| robot_arm | t_f | 9.140911746 | 9.1409 | 1.3e-06 ok | COPS robot arm; GAMS NOA / PROPT (nh = 800 discretization) | 19.5 |
| cart_pole | J (published extremal) | 58.80766746 | 58.80819 | 8.9e-06 ok | ASSET reproduction of Kelly 2017 (LGL5, 64 segments) | 24.7 |
| cart_pole | J (best of random search) | 57.9447803 | 57.9447803 | 6.4e-12 ok | independent SciPy simulation of Kelly's equations under this u(t) | 28.9 |
