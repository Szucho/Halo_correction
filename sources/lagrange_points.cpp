#include <cmath>
#include "../headers/matrix.h"
#include "../headers/lagrange_points.h"

#define SQR(x) ((x)*(x))
#define CUB(x) ((x)*(x)*(x))


/*
 lagrange_points.cpp  Lagrange point location and linearized stability
 Bertalan Szuchovszky 13.03.2026

 Implementation of functions declared in lagrange_points.h.

 Lagrange_f / Lagrange_df:
   The collinear Lagrange points L1 and L2 satisfy a quintic polynomial
   obtained by substituting the equilibrium condition (net force = 0)
   into the CR3BP equations of motion and collecting terms.
   epsilon = +1 gives the L2 equation, epsilon = -1 gives L1.
   The derivative Lagrange_df is used by newton_iter for quadratic
   convergence.

 fixpoint_stability:
   Linearizes the CR3BP equations of motion at a given collinear
   Lagrange point by evaluating the second-order partial derivatives
   of the effective potential (Oxx, Oyy, Ozz, Oxy) at that point.
   The 6x6 Jacobian of the linearized system decouples into:

     out-of-plane (z):  lambda^2 = Ozz
                        -> purely imaginary pair at L1/L2 (center)
                        -> this is the "halo" direction

     in-plane (xy):     lambda^4 + (2 - Oxx - Oyy)*lambda^2
                        + (Oxx*Oyy - Oxy^2) = 0
                        solved as a quadratic in lambda^2
                        -> at L1/L2: one real hyperbolic pair (unstable)
                           and one imaginary center pair
                        -> the real pair is the source of the
                           stable/unstable manifolds used for
                           low-energy transfers

   Returns all 6 eigenvalues as EigenPair {real, imag} representing
   lambda = real +/- i*imag. Complex square roots are handled
   explicitly to avoid std::complex overhead.

 Dependencies: matrix.h, lagrange_points.h
*/


//equation for L1, L2
double Lagrange_f(double x, double epsilon, double mu){ 
  return CUB(x)*SQR(x)+epsilon*(3-mu)*SQR(x)*SQR(x)+(3-2*mu)*CUB(x)-mu*SQR(x)-epsilon*2*mu*x-mu;
}
double Lagrane_df(double x, double epsilon, double mu){
  return 5*SQR(x)*SQR(x)+epsilon*(3-mu)*CUB(x)*4+3*(3-2*mu)*SQR(x)-mu*x*2-epsilon*2*mu;
}


std::vector<EigenPair> fixpoint_stability(const Vector & L, double mu){
  
  double x = L[0];
  double y = L[1];
  double z = L[2];

  //distance from P1 & P2
  double r1 = std::sqrt(SQR((x - mu)) + SQR(y) + SQR(z)); // P1->P3
  double r2 = std::sqrt(SQR((x + 1.0 - mu)) + SQR(y) + SQR(z)); // P2->P3

  if (r1 < 1e-14 || r2 < 1e-14) {
    throw std::invalid_argument("Division error, r1 or r2 is zero");
  }

  double r1_3 = CUB(r1);
  double r2_3 = CUB(r2);
  double r1_5 = r1_3 * SQR(r1);
  double r2_5 = r2_3 * SQR(r2);

  //second order partial derivatives of the effective potential
  double Oxx = 1 - (1-mu)/r1_3 - mu/r2_3 + 3*((1-mu)*SQR(x-mu)/r1_5 + mu*SQR(x+1-mu)/r2_5);
  double Oyy = 1 - (1-mu)/r1_3 - mu/r2_3 + 3*((1-mu)*SQR(y)/r1_5 + mu*SQR(y)/r2_5);
  double Ozz = - (1-mu)/r1_3 - mu/r2_3 + 3*((1-mu)*SQR(z)/r1_5 + mu*SQR(z)/r2_5);
  double Oxy = 3*((1-mu)*(x-mu)*y/r1_5 + mu*(x+1-mu)*y/r2_5);

  //halo (z)
  double lam_z_sq = Ozz;
  EigenPair ez1, ez2;

  if (lam_z_sq >= 0) {
    //real eigenvalue
    ez1 = { std::sqrt(lam_z_sq), 0.0};
    ez2 = {-std::sqrt(lam_z_sq), 0.0};
  } else {
    //imaginary
    ez1 = {0.0,  std::sqrt(-lam_z_sq)};
    ez2 = {0.0, -std::sqrt(-lam_z_sq)};
  }
  
  //orbital plane (xy)
  double b = 2.0 - Oxx - Oyy;
  double c = Oxx*Oyy - SQR(Oxy);
  double disc = SQR(b) - 4.0*c;
  
  auto solve_lam = [](double p_val) -> std::pair<EigenPair,EigenPair> {
    //p_val = lambda^2, extract lambda = +/-sqrt(p_val)
    if (p_val >= 0) {
      double s = std::sqrt(p_val);
      return {{ s, 0.0}, {-s, 0.0}};  //real
    } else {
      double s = std::sqrt(-p_val);
      return {{0.0, s}, {0.0, -s}};   //imaginary
    }
  };

  std::pair<EigenPair,EigenPair> p1_pair, p2_pair;
  if (disc >= 0) {
    double p1 = 0.5*(-b + std::sqrt(disc));
    double p2 = 0.5*(-b - std::sqrt(disc));
    p1_pair = solve_lam(p1);
    p2_pair = solve_lam(p2);
  } else {
    //p = (-b +/- i*sqrt(-disc))/2  complex conjugate pair
    double re = -b * 0.5;
    double im = 0.5 * std::sqrt(-disc);
    //sqrt of a complex number a+ib:
    //|p| = sqrt(re^2+im^2), sqrt(p) = sqrt((|p|+re)/2) +/- i*sqrt((|p| -re)/2)
    double mod = std::sqrt(re*re + im*im);
    double sr =  std::sqrt(0.5*(mod + re));
    double si =  std::sqrt(0.5*(mod - re));
    //4 eigenvalues: +/-(sr + i*si), +/-(sr - i*si)
    p1_pair = {{ sr,  si}, {-sr, -si}};
    p2_pair = {{ sr, -si}, {-sr,  si}};
  }

  auto [e1, e2] = p1_pair;
  auto [e3, e4] = p2_pair;

  return {e1, e2, e3, e4, ez1, ez2};
}
