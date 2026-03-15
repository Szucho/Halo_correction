#include <iostream>
#include <cmath>
#include <vector>
#include "matrix.h"


/*
 lagrange_points.h  Lagrange point location and linearized stability
 Bertalan Szuchovszky 13.03.2026

 Locates the collinear Lagrange points L1 and L2 in the CR3BP and
 computes the eigenstructure of the linearized flow at each point.

 Contents:
  - Lagrange_f / Lagrange_df:   quintic polynomial and its derivative
                                whose root gives the L1/L2 location
                                epsilon = +1 -> L2, epsilon = -1 -> L1

  - newton_iter:                Newton-Raphson root finder templated on
                                the function and derivative types
                                converges to tol=1e-14 in ~5 iterations

  - EigenPair:                  return struct {real, imag} representing
                                a complex conjugate eigenvalue pair
                                lambda = real +/- i*imag

  - fixpoint_stability:         linearizes the CR3BP equations of motion
                                at a given fixed point (Lagrange point)
                                and returns the 3 eigenvalue pairs of the
                                6x6 Jacobian as a vector<EigenPair>
                                L1/L2 have one real unstable pair,
                                one center pair, and one oscillatory pair

 Usage:
   double x0 = initial_guess;
   newton_iter(Lagrange_f, Lagrange_df, mu, +1, x0); //L2
   newton_iter(Lagrange_f, Lagrange_df, mu, -1, x0); //L1
   auto evs = fixpoint_stability({L2_x, 0.0, 0.0}, mu);

 Dependencies: matrix.h
*/


double Lagrange_f(double x, double epsilon, double mu);
double Lagrane_df(double x, double epsilon, double mu);

//Newton-iteration for Lagrange-point
template<typename FuncType>
void newton_iter(FuncType f, FuncType df,
                 double mu, double epsilon,
                 double& x0, double tol = 1e-14, 
                 double max_iter = 1000){

  for(int n = 0; n<max_iter; n++){
    double fxn = f(x0, epsilon, mu);
    if(std::abs(fxn)<tol){
      std::cout << "Found solution after " << n <<  " iterations.\n";
      return;
    }
    double dfxn = df(x0, epsilon, mu);
    if(dfxn == 0){
      throw std::invalid_argument("Zero derivative. No solution found.");
      return;
    }
    x0 -= fxn/dfxn;
  }
  throw std::invalid_argument("Exceeded maximum iterations. No solution found.");
  return;
}



struct EigenPair {
  double real, imag; //lambda = real +/- i*imag
};

std::vector<EigenPair> fixpoint_stability(const Vector & L, double mu);
