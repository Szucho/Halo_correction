#ifndef NUM_CONTINUATION_H
#define NUM_CONTINUATION_H


#include <cmath>
#include "matrix.h"
#include "diff_correction.h"

/*
 num_continuation.h Numerical continuation for halo orbits in the CR3BP
 Bertalan Szuchovszky 12.03.2026

 Implements pseudo-arclength continuation to trace families of halo orbits.
 Starting from a known corrected halo orbit, the continuation steps along
 the solution manifold using a tangent predictor followed by differential
 correction.

 Method: pseudo-arclength continuation
   ref: https://bluescarni.github.io/heyoka.py/notebooks/Pseudo%20arc-length%20continuation%20in%20the%20CR3BP.html

 State vector: [x, y, z, vx, vy, vz] (no STM needed here but I keep the shape otherwise vector size mismatch)
 Tangent vector: dy/ds computed from null space of Jacobian via Gauss elimination (2x3 with [x0,z0,vy0] space)
 Predictor:  y_pred = y_current + (dy/ds) * ds for [x0,z0,vy0]
 Corrector:  halo_pseudo_arclength_corrector which is basically the halo_differential_correction scheme
             but with one more condition on the tangent so that we can calculate the arclength
             should ONLY be used in the pseudo_arclength function (it's akin to a helper)

 Contents:
  - ContinuationResult: return struct {Matrix states, Vector periods} just the new corrected initial conditions
  - halo_pseudo_arclength_corrector: based on halo_differential_correction, modified for pseudo-arclength method
  - pseudo_arclength: continuation loop, returns family of halo orbits

 Note: z0 is the fixed parameter of the correction scheme therefore I varied [x0, z0, vy0]
       No T variation needed as T is recovered from the xz plane crossing
       I also gave integrate_to_xz a generous 2.0*current_period half time estimate
       to ensure we have a crossing - this does not affect the performance as integrate_to_xz
       uses a terminal crossing event so the integration stops after the crossing.
      
       I also left some debug statements commented out in the code, they could be useful in the future
       if i ever feel like investigating bifurcations or orbit family manifolds
       
 Dependencies: matrix.h, helpers.h, solver.h, diff_correction.h

 Yes again, it's in a header file because of the FuncType template
*/

#include "matrix.h"
#include "helpers.h"
#include "solver.h"
#include "diff_correction.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

#define SQR(x) ((x)*(x))
#define CUB(x) ((x)*(x)*(x))


struct ContinuationResult {
    Matrix states;   //each row is a corrected initial state vector
    Vector periods;  //corresponding periods
};



//WARNING! This function was only created for pseudo-arclength method
//based on the one found in diff_correction.h
//Use that for single corrections
template<typename FuncType>
DiffCorrResult halo_pseudo_arclength_corrector(Vector& y_guess,       //the predicted state (y_pred)
                                               const Vector& X_prev,  //the previous converged solution [x, z, vy]
                                               const Vector& tau,     //the tangent vector [tau0, tau1, tau2]
                                               double ds,             //step size in tangent direction
                                               double half_T_est,
                                               double h, FuncType f,
                                               double ntol = 1e-8, 
                                               double atol=1e-8, double rtol=1e-6){

  using namespace VecOps;

  //X = [x0, z0, vy0]
  Vector X = {y_guess[0], y_guess[2], y_guess[4]};

  auto F_3x3 = [&](const Vector& X_vars) -> Vector {
      Vector state = y_guess;
      state[0] = X_vars[0]; // x0
      state[2] = X_vars[1]; // z0
      state[4] = X_vars[2]; // vy0

      //symmetry conditions (vx=0, vz=0 at crossing)
      auto [t_ev, s_ev] = integrate_to_xz(state, half_T_est, h, f, atol, rtol);
      
      //pseudo-arclength constraint
      //(X-X_prev)*tau-ds=0
      double constraint = (X_vars[0] - X_prev[0]) * tau[0] +
                          (X_vars[1] - X_prev[1]) * tau[1] +
                          (X_vars[2] - X_prev[2]) * tau[2] - ds;

      return {s_ev[3], s_ev[5], constraint};
  };

  newton_gauss_3x3(F_3x3, X, ntol);

  Vector corrected = y_guess;
  corrected[0] = X[0];
  corrected[2] = X[1];
  corrected[4] = X[2];
  
  auto [final_t, final_s] = integrate_to_xz(corrected, half_T_est, h, f, atol, rtol);
  
  return {2.0 * final_t, corrected};
}




template<typename FuncType>
ContinuationResult pseudo_arclength(FuncType f,
                                     const Vector& state,   //corrected initial state (with STM so 42 dim)
                                     double period,         //corrected initial period
                                     double mu,             //CR3BP mass parameter
                                     double ds,             //arclength step size
                                     double h,              //integrator initial step size
                                     int max_iter,          //number of continuation steps
                                     double ntol = 1e-8,    //Newton-Gauss iteration tolerance
                                     double atol = 1e-8,    //integrator absolute tolerance
                                     double rtol = 1e-6) {  //integrator relative - || -
  using namespace VecOps;
  const int state_dim = 6;
  const double eps = 1e-6;
  

  Matrix sol_states(1, state_dim);
  Vector state6(state.begin(), state.begin()+state_dim);
  sol_states.setRow(0, state6);
  Vector sol_periods;
  sol_periods.push_back(period);

  Vector y_current = state;
  double current_period = period;

  double prev_tau0 = 0.0, prev_tau1 = 1.0, prev_tau2 = 0.0;  //initial direction: z0 as its the fixed parameter
                                                             //of my halo differential correction scheme

  double tau1_sign = 0.0; //sign of tau1, double as we will multiply it by tau1 which is a double
  for (int k = 0; k < max_iter; k++) {
    // std::cout << "Continuation step " << k << "\n";

    //current point in [x0, z0, vy0] space
    Vector X_current = {y_current[0], y_current[2], y_current[4]};
    //relative perturbation
    const double eps_x  = eps * std::max(std::abs(X_current[0]), 1.0);
    const double eps_z = eps * std::max(std::abs(X_current[1]), 1.0);
    const double eps_vy  = eps * std::max(std::abs(X_current[2]), 1.0);

    // std::cout <<"eps_x: "  << eps_x << "\t"
    //           <<"eps_z: "  << eps_z << "\t"
    //           <<"eps_vy: " << eps_vy  << std::endl;

    auto eval_F = [&](double x0, double z0, double vy0) -> Vector {
      Vector state_guess = y_current;
      state_guess[0] = x0;
      state_guess[2] = z0;
      state_guess[4] = vy0;
      //integrate past initial xz plane, then find crossing
      auto res = integrate_to_xz(state_guess, 2.0*current_period, h, f, atol, rtol); //generous T/2 estimate to
      return {res.half_state[3], res.half_state[5]};                                 //ensure we have xz crossing
    };                                                                               //this doesn't tank performance
                                                                                     //as xz crossing is terminal

    //base evaluation (the current already corrected state)
    Vector F0 = eval_F(X_current[0], X_current[1], X_current[2]);
    //2x3 Jacobi DF (forward differences)
    //col 0: F perturbed in x0
    Vector F_dx = eval_F(X_current[0] + eps_x, X_current[1], X_current[2]);
    //col 1: F perturbed in z0
    Vector F_dz = eval_F(X_current[0], X_current[1] + eps_z, X_current[2]);
    //col 2: F perturbed in vy0
    Vector F_dv = eval_F(X_current[0], X_current[1], X_current[2] + eps_vy);

    //DF rows are DF[i][j] = dFi/dXj a simple Jacobi
    //here come the forward differences
    double DF00 = (F_dx[0] - F0[0])/eps_x;    //dF1/dx0
    double DF01 = (F_dz[0] - F0[0])/eps_z;    //dF1/dz0
    double DF02 = (F_dv[0] - F0[0])/eps_vy;   //dF1/dvy0

    double DF10 = (F_dx[1] - F0[1])/eps_x;    //dF2/dx0
    double DF11 = (F_dz[1] - F0[1])/eps_z;   //dF2/dz0
    double DF12 = (F_dv[1] - F0[1])/eps_vy;   //dF2/dvy0

    //tangent = null space of DF (2x3) = cross product of rows
    double tau0 = DF01*DF12 - DF02*DF11; //x0
    double tau1 = DF02*DF10 - DF00*DF12; //z0
    double tau2 = DF00*DF11 - DF01*DF10; //vy0

    double tau_norm = std::sqrt(tau0*tau0 + tau1*tau1 + tau2*tau2);
    if (tau_norm < 1e-14)
      throw std::runtime_error("Zero tangent vector in continuation");
    tau0 /= tau_norm;
    tau1 /= tau_norm;
    tau2 /= tau_norm;
    if (k == 0) {
      //align with ds sign
      //just ensure tau1 (z0 direction) has the same sign as ds otherwise oscillation occurs
      if (tau1 < 0) { 
        tau0 = -tau0; tau1 = -tau1; tau2 = -tau2; 
      }
      tau1_sign = (tau1 > 0.0) ? 1.0 : -1.0; //1 if positive, -1 if negative
    } else {
      //align with previous tangent
      double dot = tau0*prev_tau0 + tau1*prev_tau1 + tau2*prev_tau2;
      if (dot < 0) { tau0=-tau0; tau1=-tau1; tau2=-tau2; }
    }

    //if after a turning point (sign change in tau1~z0) we continue the continuation,
    //we get orbits that we've already found but now we are going backwards on the manifold
    //If we start with an arbitrary orbit on the manifold, we should have a -ds and ds so that 
    //we trace the manifold in both directions -> generate the whole orbit family
    //Some interesting ideas that I might implement later (I think both of these were done already):
    //  -> Properties of the manifold of an orbital family
    //  -> Bifurcation analysis

    //We want to step in the direction of ds and then for
    //subsequent steps we just follow the previous tangent direction.
    prev_tau0 = tau0;
    prev_tau1 = tau1;
    prev_tau2 = tau2;

    // std::cout << "DF  = [" << DF00 << " " << DF01 << " " << DF02 << "]\n";
    // std::cout << "      [" << DF10 << " " << DF11 << " " << DF12 << "]\n";
    // std::cout << "tau = [" << tau0 << " " << tau1 << " " << tau2 << "]\n";
    // std::cout << "F0  = [" << F0[0] << " " << F0[1] << "]\n";

    //predictor step along tangent in [x0, vy0, T] space
    Vector y_pred = y_current;
    y_pred[0]     = X_current[0] + tau0 * ds;  //x0
    y_pred[2]     = X_current[1] + tau1 * ds;  //z0
    y_pred[4]     = X_current[2] + tau2 * ds;  //vy0
    
    if (k==0){
      std::cout << "k=0 tau=["<<tau0<<","<<tau1<<","<<tau2<<"] ds="<<ds<<"\n";
      std::cout << "Predicted: x0="<<y_pred[0]<<" z0="<<y_pred[2]<<" vy0="<<y_pred[4]<<"\n";
    }

    // std::cout << "Predicted:\nx0 = " << y_pred[0]
    //           << "\nz0 = " << y_pred[2]
    //           << "\nvy0 = " << y_pred[4]<<std::endl;

    //corrector, here 2*T is used instead of T/2 (which is the half period estimate) 
    //to ensure that we really have a xz plane crossing
    //the crossing event is terminal in the halo_diff_corr
    //func so performance wise this is not an issue
    Vector tau = {tau0, tau1, tau2};
    DiffCorrResult corr = halo_pseudo_arclength_corrector(y_pred, X_current, 
                                                          tau, ds,
                                                          2.0*current_period, 
                                                          h, f, ntol, atol, rtol);

    //to store 6 component physical state vector
    Vector corr6(corr.corrected_state.begin(), corr.corrected_state.begin()+state_dim);
    //store
    sol_states.resize(k + 2, state_dim);
    sol_states.setRow(k + 1, corr6);
    sol_periods.push_back(corr.period);

    y_current = corr.corrected_state;
    current_period = corr.period;
  }

  return {sol_states, sol_periods};
}


#endif //NUM_CONTINUATION_H
