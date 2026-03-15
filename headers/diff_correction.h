#ifndef DIFF_CORRECTION_H
#define DIFF_CORRECTION_H

#include "matrix.h"
#include "solver.h"
#include <stdexcept>
#include <iostream>

#define SQR(x) ((x)*(x))
#define CUB(x) ((x)*(x)*(x))

/*
 diff_correction.h Differential correction for halo orbits in the CR3BP
 Bertalan Szuchovszky 12.03.2026

 Implements the differential correction procedure for finding periodic
 halo orbits in the Circular Restricted Three-Body Problem (CR3BP).
 The corrector exploits the symmetry of halo orbits about the xz-plane:
 a periodic orbit must cross the xz-plane (y=0) perpendicularly with vx=vz=0 @ y=0.

 State vector: [x, y, z, vx, vy, vz, STM(36)]
   indices: 0=x, 1=y, 2=z, 3=vx, 4=vy, 5=vz, 6..41=STM (flattened, row-major)

 Correction variables: X = [x0, vy0]
 Correction conditions: F(X) = [vx, vz] = 0 at xz-plane crossing (y=0)

 Contents:
  - event_xz:            event function y[1]=0 for solve_de

  - IntToxzResult:       return struct for integrate_to_xz

  - integrate_to_xz:     integrates until first xz-plane crossing
                         t0=1e-8 to skip initial condition crossing,
                         y0 is a guess so this is fine, the corrector
                         will find a solution regardless

  - newton_gauss_2x2:    Gauss-Newton root finding for 2x2 system
                         numerical Jacobian by forward differences (eps=1e-8)
                         2x2 linear system (J^T*J)dX = J^T*F 
                         solved analytically (J^T is transpose of J)
                         WARNING: costs 3x integrate_to_xz per iteration
                         but I couldn't find a better solution

  - newton_gauss_3x3     Gauss-Newton root finding for 3x3 system
                         works on the same principle as the 2x2 one
                         needed for pseudo-arclength continuation
                         specifically for halo_pseudo_arclength_corrector
                         found in num_continuation.h

  - DiffCorrResult:      return struct for halo_differential_correction

  - halo_differential_correction: full correction procedure returns 
                                  corrected initial state vec and period T

 Usage:
   DiffCorrResult res = halo_differential_correction(y0, T_half_estimate, h, f);
   res.period           -> corrected period T
   res.corrected_state  -> corrected initial state [x0,0,z0,0,vy0,0,STM]

 Dependencies: matrix.h, solver.h
*/



//event when object crosses the xz plane (x, y=0, z, ....)
inline double event_xz(double t, const Vector& y) {
    return y[1];  //y component of position
}

struct IntToxzResult{
  double half_time;  //time at T/2 based on xz-plane crossing
  Vector half_state; //state vector at T/2 - || - 
};

template<typename FuncType>
IntToxzResult integrate_to_xz(const Vector& y0,
                              double half_period_estimate,
                              double h, FuncType f,
                              double atol = 1e-8,
                              double rtol = 1e-6) {
  using namespace VecOps;

  SolverResult res = solve_de(1e-6, half_period_estimate, y0, //we can use t0=1e-8 as y0 is just a guess here and
                              f, event_xz, true,              //the corrector will find the solution either ways
                              h, atol, rtol);
  
  if (res.event_times.size()==0) throw std::runtime_error("No xz-plane crossing found within estimated half period");
  Vector state_at_event = res.event_states.row(0);
  return {res.event_times[0], state_at_event};
}


template<typename FuncType>
void newton_gauss_2x2(FuncType F, Vector& X, double ntol = 1e-8, int max_iter = 100) {
/*
  Sadly this guy calls F 3 times (once for Fval, dF0 and dF1) and each F call runs a full integrate_to_xz.
  Each iteration costs 3 integrations (if well conditioned problem: ~5 N-G iterations = 15 integrations!!!). 
  This seems unavoidable for a numerical Jacobian just worth keeping in mind when going for performance.
  If YOU have a better solution message me :D
                  ___________________________
                  \###   ####   ####   #### /
                   ###   ####   ####   ####|
                   ###   ####   ####   ####|
                   ###   ####   ####   ####|
                   ###   ####   ####   ####|
                   ###   ####   ####   ####|
                   ###   ####   ####   ####|
                   ###   ####   ####   ####|
                   @@@@@@@@@@@@@@@@@@@@@@@@@
                   @@@@@ @@@@@@@@@@@ @@@@@@@
                   @@@@   @@@@@@@@@   @@@@@@
                   @         @@            @
                   @@@  @  @@@@@@   @   @@@@
                   @@  @@@  @@@@   @@@   @@@
                   @@@@@@@@@@@@@@@@@@@@@@@@@
       (~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~)
        (:::::::::::::::::::::::::::::::::::::::::::::)
            888 888::{*****xxx   xxx******}@@@@8 88
           888 88 8    ( 0 )xxx/\ xx( 0 ) @@@@@@@ 8
            88 888 8     -   xx  |    -      @@@   88
            8 888 88          x   |         @@@@@@ 8
           8 8 888/8          x    |          888  888
            \88/  888        C     o)        8/  @@
             \/@ @8/ \         \_/         /@@@@@  88
                @     |     /         \   |   @@@
                     / |     (mmmmmmm)   |   \
  _________________ /\  \_     (wwww)  _/ /   \
 /                    \    XXX   v  XXX  /     \___________
/                 /    \    xxxxxxxxxx  /       /          \
|                /      \    xxxxxxxx  /      /             \
|               /        \    xxxxxx  /      /              |
|               \         \    xxxx  /     /                |
|                 \        \    xx  /     /                 |
|                   \       \       |    /                  |
|                     \      \ @@@@@@   /                   |
|          /@           \     \@@@@@@  /                    |
|         / /_           \    |@@@@@@ /                     |
|        /|/ _)            \ /@@@@@@@                       |
|       (-----;             |@@@@@@@@@@                     |
  V    /(---"              |@@@@@@@@@@@@                    |
|  \   /   )                |@@@@@@@@@@                     |
|    ---------              |@@@@@@@@@@                     |
|   /       /               |@@@@@@@@@@                     |
|  /       /                |@@@@@@@@@@                     |
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
------------------------------------------------
This ASCII pic can be found at
https://asciiart.website/art/6281
  */ 
  using namespace VecOps;
  const double eps = 1e-8;

  for (int iter = 0; iter < max_iter; iter++) {
    Vector Fval = F(X);

    //check convergence on residual
    double Fnorm = std::sqrt(Fval[0]*Fval[0] + Fval[1]*Fval[1]);
    if (Fnorm < ntol) return;

    //numerical Jacobi-matrix J (2x2) by forward differences
    //J[:,0] = (F(X + eps*e0) - F(X)) / eps forward derivative
    //J[:,1] = (F(X + eps*e1) - F(X)) / eps - || -
    Vector X0 = X; X0[0] += eps;
    Vector X1 = X; X1[1] += eps;
    Vector dF0 = F(X0);
    Vector dF1 = F(X1);

    //J cols
    double J00 = (dF0[0] - Fval[0]) / eps;
    double J10 = (dF0[1] - Fval[1]) / eps;
    double J01 = (dF1[0] - Fval[0]) / eps;
    double J11 = (dF1[1] - Fval[1]) / eps;

    //A = J^T * J  (2x2) matrix
    double A00 = J00*J00 + J10*J10;
    double A01 = J00*J01 + J10*J11;
    double A10 = J01*J00 + J11*J10;  // =A01 since J^T*J is symmetric
    double A11 = J01*J01 + J11*J11;

    //b = J^T * F  (2x1) vector
    double b0 = J00*Fval[0] + J10*Fval[1];
    double b1 = J01*Fval[0] + J11*Fval[1];

    //solve A*dX = b analytically
    double det = A00*A11 - A01*A10;
    if (std::abs(det) < 1e-14) //check for singularities first
        throw std::runtime_error("Singular Jacobian in Gauss-Newton");

    double dX0 = ( A11*b0 - A01*b1) / det;
    double dX1 = (-A10*b0 + A00*b1) / det;

    //update next step X_{n+1} = X_n - dX
    double alpha = 1.0; //damping
    for(int ls = 0; ls < 10; ls++) { //backtracking linesearch
        Vector F_new = F({X[0] - alpha*dX0, X[1] - alpha*dX1});
        double Fnorm_new = std::sqrt(F_new[0]*F_new[0] + F_new[1]*F_new[1]);
        if(Fnorm_new < Fnorm) break;
        alpha *= 0.5;
    }
    //update with damped step
    X[0] -= alpha * dX0;
    X[1] -= alpha * dX1;

    //check convergence on step size
    double dXnorm = std::sqrt((alpha*dX0)*(alpha*dX0) + (alpha*dX1)*(alpha*dX1));
    if (dXnorm < ntol) return;
  }
  throw std::runtime_error("Gauss-Newton did not converge, max iterations reached");
}


//same but for 3x3 needed for pseudo-arclength later on but I will write this here too
template<typename FuncType>
void newton_gauss_3x3(FuncType F, Vector& X, double ntol = 1e-8, int max_iter = 100) {
  using namespace VecOps;
  const double eps = 1e-8;

  for (int iter = 0; iter < max_iter; iter++) {
      Vector Fval = F(X); //returns {res1, res2, res3}
      
      double Fnorm = std::sqrt(SQR(Fval[0]) + SQR(Fval[1]) + SQR(Fval[2]));
      if (Fnorm < ntol) return;

      //3x3 Jacobi matrix again via forward differences
      Vector X0 = X; X0[0] += eps;
      Vector X1 = X; X1[1] += eps;
      Vector X2 = X; X2[2] += eps;

      Vector F_dx = F(X0);
      Vector F_dz = F(X1);
      Vector F_dv = F(X2);

      //J cols
      double J00 = (F_dx[0] - Fval[0]) / eps;
      double J10 = (F_dx[1] - Fval[1]) / eps;
      double J20 = (F_dx[2] - Fval[2]) / eps;

      double J01 = (F_dz[0] - Fval[0]) / eps;
      double J11 = (F_dz[1] - Fval[1]) / eps;
      double J21 = (F_dz[2] - Fval[2]) / eps;

      double J02 = (F_dv[0] - Fval[0]) / eps;
      double J12 = (F_dv[1] - Fval[1]) / eps;
      double J22 = (F_dv[2] - Fval[2]) / eps;

      //J*dX=Fval using Cramers rule
      //we basically want to find dX such that X_new=X-dX
      double det = J00 * (J11 * J22 - J12 * J21) -
                   J01 * (J10 * J22 - J12 * J20) +
                   J02 * (J10 * J21 - J11 * J20);

      if (std::abs(det) < 1e-15) 
          throw std::runtime_error("Singular Jacobian in Gauss-Newton 3x3");

      //numerators for Cramers rule (replacing columns with Fval)
      auto get_det = [](double a, double b, double c, 
                        double d, double e, double f, 
                        double g, double h, double i) {
          return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);
      };

      double dX0 = get_det(Fval[0], J01, J02, Fval[1], J11, J12, Fval[2], J21, J22) / det;
      double dX1 = get_det(J00, Fval[0], J02, J10, Fval[1], J12, J20, Fval[2], J22) / det;
      double dX2 = get_det(J00, J01, Fval[0], J10, J11, Fval[1], J20, J21, Fval[2]) / det;

      //simple backtracking line search agagain
      double alpha = 1.0; //damping
      for(int ls = 0; ls < 8; ls++) {
          Vector X_try = {X[0] - alpha*dX0, X[1] - alpha*dX1, X[2] - alpha*dX2};
          Vector F_new = F(X_try);
          double Fnorm_new = std::sqrt(SQR(F_new[0]) + SQR(F_new[1]) + SQR(F_new[2]));
          
          if(Fnorm_new < Fnorm) break;
          alpha *= 0.5;
      }

      X[0] -= alpha * dX0;
      X[1] -= alpha * dX1;
      X[2] -= alpha * dX2;

      if (std::sqrt(SQR(alpha*dX0) + SQR(alpha*dX1) + SQR(alpha*dX2)) < ntol) return;
  }
  throw std::runtime_error("Newton 3x3 failed to converge");
}



struct DiffCorrResult {
    double period;
    Vector corrected_state;
};

template<typename FuncType>
DiffCorrResult halo_differential_correction(Vector& y0,
                                            double half_period_estimate,
                                            double h,
                                            FuncType f,
                                            double ntol = 1e-8,
                                            double atol = 1e-8,
                                            double rtol = 1e-6) {
  using namespace VecOps;

  double event_time = 0.0;

  //F([x0, vy0]) = [vx, vz] at xz-plane crossing ~> drive these to zero OR ELSE >:[
  auto correction_conditions = [&](const Vector& vars) -> Vector {
    Vector state = y0;
    state[0] = vars[0];  // x0
    state[4] = vars[1];  // vy0

    auto [t_event, state_at_event] = integrate_to_xz(state,
                                                      half_period_estimate,
                                                      h, f, atol, rtol);
    event_time = t_event;
    return {state_at_event[3],   // vx = 0
            state_at_event[5]};  // vz = 0
  };

  //initial guess X = [x0, vy0]
  Vector X = {y0[0], y0[4]};

  newton_gauss_2x2(correction_conditions, X, ntol);

  //apply corrected values back to state
  Vector corrected = y0;
  corrected[0] = X[0];  //corrected x0
  corrected[4] = X[1];  //corrected vy0
  double period = 2.0 * event_time; //2*T/2

  // std::cout << "Differential correction converged. T = " << period << std::endl;
  return {period, corrected};
}

#endif //DIFF_CORRECTION_H
