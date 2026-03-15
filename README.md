# Halo_correction
Halo differential correction method, my code for my BSc thesis (ELTE TTK physics) revised (optimalized): "Numerical generation of cislunar NRHO-s and stability analysis"
Disclaimer: I wrote this code alone (ackownledged by my supervisor Dr Áron Süli in the grade recommendation letter) so it's not perfect & as optimized as it could be because I'm not a programmer


## Pipeline

Starting from an initial guess, the code:

1. Calculates mu mass parameter from m1,m2 masses of primaries and then locates L1/L2 via Newton iteration on the collinear Lagrange point quintic
2. Corrects the guess to a true periodic halo orbit via differential correction exploiting xz-plane symmetry: vx = vz = 0 at y = 0 so this is the time variational differential correction method
3. Traces the full halo family using pseudo-arclength continuation (which uses a modified version of the halo differential correction taylored for the continuation)
4. Computes for each orbit:
   - Symplectic stability indices $\nu_1$, $\nu_2$ from the monodromy matrix
   - Maximal Lyapunov exponent from the monodromy eigenvalues
   - Jacobi constant (post-process in python script)
   - Perilune / apolune distances (post-process in python script)
5. Writes trajectories and family summary to CSV for Python post-processing

## Example results

The generated family spans from L2 near-rectilinear halo orbits (NRHOs) 
close to the Moon to large-amplitude halo orbits, covering perilune 
distances from ~500 km to ~30,000 km in the Earth-Moon system.
Stability indices and Lyapunov exponents are consistent with results 
reported in the literature: Zhang, Renyong. "A review of periodic orbits in the circular restricted three-body problem." Journal of Systems Engineering and Electronics 33.3 (2022): 612-646.
Check the "nrho" folder to see the figures and csv files

## Numerical methods

| **Component** | **Method** |
|---|---|
| ODE integration | Dormand-Prince 5(4) with adaptive step size control |
| Event detection | Brent root finding on linearly interpolated steps |
| Differential correction | Gauss-Newton iteration (2×2 system) |
| Continuation | Pseudo-arclength with tangent predictor (solve 3x3 system with modified halo differential correction function) |
| Lyapunov exponents | QR factorization, generally run integration for 3 periods for convergence |
| Stability indicies | STM propagation, eigenvalues of monodromy matrix |

## Build
```bash
clang++ -std=c++20 -O2 -I./headers sources/helpers.cpp \
        sources/lagrange_points.cpp -o builds/corrector sources/corrector.cpp
```
The day will come when I will have to write a CMake file but it's not today.

## Notes

The code uses a custom `Matrix` / `Vector` implementation (`matrix.h`) 
with operator overloading for clean numpy-style arithmetic in the 
integrator and corrector routines, avoiding any external linear algebra 
dependency. Templates are used throughout for the ODE right-hand side 
and event functions, which is why most of the numerical machinery lives 
in header files (I couldn't make it work otherwise :/).

## References

- Dormand & Prince (1980) — A family of embedded Runge-Kutta formulae
- Dieci, Russell & Van Vleck (1997) — On the computation of Lyapunov 
  exponents for continuous dynamical systems
- Zhang R. (2022) — A review of periodic orbits in the circular 
  restricted three-body problem
- Brent (1973) — Algorithms for Minimization Without Derivatives
- MINPACK (Argonne National Laboratory, 1980) — qform, qrfac


and a lot more cited in my BSc thesis :)
