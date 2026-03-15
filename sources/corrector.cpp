#include <cmath>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include "../headers/matrix.h"
#include "../headers/solver.h"
#include "../headers/diff_correction.h"
#include "../headers/num_continuation.h"
#include "../headers/lagrange_points.h"


/*
 corrector.cpp  CR3BP halo orbit family generator (my BSc work after some revisions)
 Bertalan Szuchovszky 15.03.2026

 Computes a family of L2 halo orbits in the Earth-Moon Circular Restricted
 Three-Body Problem (CR3BP) via differential correction and pseudo-arclength
 continuation, then analyses the orbital stability of each family member.

 The pipeline:
   1. Lagrange points  - L1/L2 locations and linearized eigenstructure
                         via Newton iteration (lagrange_points.h)
   2. CR3BP + STM      - equations of motion augmented with the 6x6 State
                         Transition Matrix for variational analysis
   3. Diff. correction - corrects an initial halo guess to a true periodic
                         orbit by driving vx=vz=0 at the xz-plane crossing
                         (diff_correction.h, solver.h)
   4. Continuation     - pseudo-arclength continuation traces the full halo
                         family from the corrected seed orbit
                         (num_continuation.h)
   5. Stability        - monodromy matrix (STM after one period) gives
                         symplectic stability indices nu1, nu2 and the
                         maximal Lyapunov exponent for each orbit
   6. Output           - trajectories and family summary written to CSV
                         for post-processing in Python

 Build (I WILL NOT WRITE A CMAKE):
   clang++ -std=c++20 -O2 -I./headers sources/helpers.cpp \
           sources/lagrange_points.cpp -o builds/corrector ./sources/corrector.cpp

 Output files:
   nrhos/trajectories.csv  - full trajectory of each orbit (orbit_id, t, x,y,z,vx,vy,vz)
   nrhos/family.csv        - after continuation + correction orbit summary (x0,z0,vy0,T,nu1,nu2,lambda_max)

 Dependencies: matrix.h, solver.h, helpers.h, diff_correction.h,
               num_continuation.h, lagrange_points.h
 
 Note: there are TWO (2) differential correction algorithms that we can use:
   - halo_differential_correction in diff_correction.h
   - halo_pseudo_arclength_corrector in num_continuation.h
 The second one is only needed for the numerical continuation and uses 3 residuals so that 
 the next solution will stay on the tangent line in the pseudo-arclength continuation method.
 If You want to correct a single orbit just use the one in diff_correction.h it's made for that purpose
 The other one in num_continuation.h should ONLY and ONLY be used in the pseudo-arclength method,
 I don't recommend trying to use it here in main. You have been warned, proceed at your own risk
*/





using namespace VecOps;

//macros
#define SQR(x) ((x)*(x))
#define CUB(x) ((x)*(x)*(x))



double mu_func(double m1, double m2){
    if (m1 < 0 || m2 < 0) { //NO negative mass thank you
        throw std::invalid_argument("Masses must be non-negative."); 
    }
    if (m1 + m2 == 0) { //NO zero mass
        throw std::invalid_argument("Sum of masses must be greater than zero.");
    }
    return std::min(m1, m2)/(m1+m2); //mu = m2/(m1+m2), with m2<m1
}



Vector crtbp_with_STM(double t, const Vector& state_ext, double mu) {
  int state_dim = 6;      // Dimension of the state

  //state vector vals
  double x = state_ext[0];
  double y = state_ext[1];
  double z = state_ext[2];
  double u = state_ext[3];
  double v = state_ext[4];
  double w = state_ext[5];

  //distance from P1 & P2
  double r1 = std::sqrt(SQR((x - mu)) + SQR(y) + SQR(z)); // P1->P3
  double r2 = std::sqrt(SQR((x + 1.0 - mu)) + SQR(y) + SQR(z)); // P2->P3

  //DONT enter the singularities or I will cry and so will the code now
  if (r1 < 1e-14 || r2 < 1e-14) {
    throw std::invalid_argument("Division error, r1 or r2 is zero");
  }

  //equations of motion
  double dxdt = u;
  double dydt = v;
  double dzdt = w;
  double dudt = 2.0 * v + x - ((1.0 - mu) * (x - mu)) / CUB(r1) - mu * (x + 1.0 - mu) / CUB(r2);
  double dvdt = -2.0 * u + y * (1.0 - (1.0 - mu) / CUB(r1) - mu / CUB(r2));
  double dwdt = -z * ((1.0 - mu) / CUB(r1) + mu / CUB(r2));

  double r1_5 = CUB(r1) * SQR(r1);
  double r2_5 = CUB(r2) * SQR(r2);

  //Jacobi matrix - block matrix
  Matrix J(state_dim, state_dim, 0.0);

  //the humble identity matrix - the upper right block in our Jacobi block matrix
  J(0,3) = 1.0; 
  J(1,4) = 1.0;  
  J(2,5) = 1.0;

  //partial derivatives of dudt
  J(3,0) = 1.0-(1.0-mu)/CUB(r1)-mu/CUB(r2)+3.0*((1-mu)*SQR(x-mu)/r1_5+mu*SQR((x+1.0-mu))/r2_5);
  J(3,1) = 3.0*((1-mu)*(x-mu)*y/r1_5 + mu*(x+1.0-mu)*y/r2_5);
  J(3,2) = 3.0*((1-mu)*(x-mu)*z/r1_5 + mu*(x+1.0-mu)*z/r2_5);
  J(3,4) = 2.0;  

  //partial derivatives of dvdt
  J(4,0) = 3.0*((1-mu)*(x-mu)*y/r1_5 + mu*(x+1.0-mu)*y/r2_5);
  J(4,1) = 1.0-(1.0-mu)/CUB(r1)-mu/CUB(r2) + 3.0*(((1.0-mu)*SQR(y))/r1_5+ mu*SQR(y)/r2_5);
  J(4,2) = 3.0*((1.0-mu)*y*z/r1_5 + mu*y*z/r2_5);
  J(4,3) = -2.0;

  // Partial derivatives of dwdt
  J(5,0) = 3.0*((1-mu)*(x-mu)*z/r1_5 + mu*(x+1.0-mu)*z/r2_5);
  J(5,1) = 3.0*((1.0-mu)*y*z/r1_5 + mu*y*z/r2_5);
  J(5,2) = -(1.0-mu)/CUB(r1)-mu/CUB(r2) + 3.0*((1.0-mu)*SQR(z)/r1_5 + mu*SQR(z)/r2_5);

  //compute dPhi/dt = J * Phi
  Matrix dPhi(state_dim, state_dim, 0.0);
  for (int i = 0; i < state_dim; i++) {
    for (int k = 0; k< state_dim; k++){
      for (int j = 0; j<state_dim; j++){
        //Phi is stored in state_ext starting at index 6
        //assuming it was passed in column major
        dPhi(i, j) += J(i,k) * state_ext[6 + j * 6 + k];
      }
    }
  }
  //concatenate derivatives into a single output vector
  Vector derivs = {dxdt, dydt, dzdt, dudt, dvdt, dwdt};
  for (int j = 0; j < state_dim; j++){ //COLUMN MAJOR ORDER NEEDED FOR QR funcs in MINPACK!!!+
    for (int i = 0; i < state_dim; i++){
      derivs.push_back(dPhi(i,j));
    }
  }
  return derivs;
}

/* :::::::::::::::::::::::::::::::::..
  orbital stability analysis happens here
*/

//stability indices: nu_+ and nu_- based on 
//Zhang Ryong: "A review of periodic orbits in the circular restricted three-body problem"
std::pair<double,double> stability_indices(const SolverResult& res, size_t row) {
  //state after a period
  Vector state_full(res.solution.cols());
  for(int j = 0; j < res.solution.cols(); j++)
    state_full[j] = res.solution(row, j);

  //get monodromy matrix = STM(T,0)
  Matrix M(6,6);
  for(int j=0;j<6;j++)
    for(int i=0;i<6;i++)
      M(i,j) = state_full[6 + j*6 + i];

  // //transpose for check
  // Matrix MT(6,6);
  // MT = M.transpose();

  // //debug
  // Matrix Omega(6,6,0.0);
  // Omega(0,3) =  1.0; Omega(3,0) = -1.0;
  // Omega(1,4) =  1.0; Omega(4,1) = -1.0;
  // Omega(2,5) =  1.0; Omega(5,2) = -1.0;
  // Omega(0,1) = -2.0; Omega(1,0) =  2.0;
  //
  // Matrix symp = MT*Omega*M-Omega;
  // 
  // double err = 0.0;
  // for(int i=0;i<6;i++)
  // for(int j=0;j<6;j++)
  //     err = std::max(err, std::abs(symp(i,j)));
  // 

  //print
  // std::cout << "Symplectic error = " << err << std::endl;

  // auto det_check = [](Matrix A) -> double {
  //   int n = 6;
  //   double det = 1.0;
  //   for(int k = 0; k < n; k++) {
  //       int maxRow = k;
  //       for(int i = k+1; i < n; i++)
  //           if(std::abs(A(i,k)) > std::abs(A(maxRow,k))) maxRow = i;
  //       if(maxRow != k) {
  //           for(int j = 0; j < n; j++) std::swap(A(k,j), A(maxRow,j));
  //           det *= -1.0;
  //       }
  //       if(std::abs(A(k,k)) < 1e-14) return 0.0;
  //       det *= A(k,k);
  //       for(int i = k+1; i < n; i++) {
  //           double f = A(i,k)/A(k,k);
  //           for(int j = k; j < n; j++)
  //               A(i,j) -= f*A(k,j);
  //       }
  //   }
  //   return det;
  // };
  //
  // std::cout << "det(M) = " << det_check(M) << " (should be 1.0)\n";

  //trace of the monodromy matrix
  double trM = 0.0;
  for(int i=0;i<6;i++) trM += M(i,i);
  std::cout << "tr(M) = " << trM << "\n";
  //trace of M^2
  double trM2 = 0.0;
  for(int i=0;i<6;i++)
    for(int k=0;k<6;k++)
      trM2 += M(i,k)*M(k,i);
  
  double disc = std::sqrt(std::max(0.0, 8.0 + 2.0*trM2 - trM*trM)); //no complex discriminants
  double nu1 = 0.25*(trM + disc);                                   //if complex -> no real part -> stable :D
  double nu2 = 0.25*(trM - disc);
  return {nu1, nu2};
}

void Lyapunov_exp(double& cumulative_log, const double& total_time) {
  cumulative_log = cumulative_log / total_time;
}

/* ::::::::::::::::::::::::::::::::.
  different section for file writing and
  saving trajectories, orbits, stab indices ...
*/


//issue with the continuation: we get physical initial conditions
//but we still need to add the stm (identity matrix) to them
//in order to integrate the augmented crtbp equation which
//allows us to caclulate either the stability indices
//based on the monodromy matrix (STM after an orbital period)
//or the Lyapunov exponents via QR factorization
Vector extend_with_STM(const Matrix& states, size_t row) {
  Vector y(42, 0.0);
  for(int j = 0; j < 6; j++) y[j] = states(row, j);
  //identity STM
  for(int j = 0; j < 6; j++) y[6 + j*6 + j] = 1.0;
  return y;
}


void write_trajectories(const std::vector<SolverResult>& results, const std::string& filename) {
  std::ofstream f(filename);
  f << std::setprecision(15);
  f << "orbit_id,t,x,y,z,vx,vy,vz\n";
  for(size_t i = 0; i < results.size(); i++) {
    for(size_t row = 0; row < results[i].solution.rows(); row++) {
      f << i << ","
        << results[i].times[row] << ",";
      for(int j = 0; j < 6; j++)
        f << results[i].solution(row,j) << (j<5 ? "," : "\n");
    }
  }
}


//write full family summary
void write_family(const ContinuationResult& family,
                  const std::vector<std::pair<double,double>>& nu,
                  const std::vector<double>& lambda_max,
                  const std::string& filename,
                  double mu, double L1_x, double L2_x) {

  std::ofstream f(filename);
  if(!f) throw std::runtime_error("Cannot open " + filename);
  f << "# mu=" << std::setprecision(15) << mu
    << " L1_x=" << L1_x << " L2_x=" << L2_x <<"\n";
  f << "orbit_id,x0,y0,z0,vx0,vy0,vz0,T,nu1,nu2,lambda_max\n";
  for(size_t i=0; i < family.periods.size(); i++) {
    f << i << ",";
    for(int j=0;j<6;j++)
      f << family.states(i,j) << ",";
    f << family.periods[i] << ","
      << nu[i].first << ","
      << nu[i].second << ","
      << lambda_max[i] << "\n";
  }
}



int main(){
  //mu
  double m2 = 0.07346e24; //Moon mass
  double m1 = 5.9724e24;  //Earth mass
  double mu  = mu_func(m1,m2);
  std::cout << "mu: " << mu << std::endl;

  //Lagrange points
  double L1x0 = (mu/(3.0-2.0*mu)); 
  double L2x0 = (-mu/(3.0+2.0*mu));
  newton_iter(Lagrange_f, Lagrane_df, mu, 1, L1x0);
  newton_iter(Lagrange_f, Lagrane_df, mu, -1, L2x0);
  double L2_x = mu-1-L1x0; //epsilon = 1 -> L2
  double L1_x = mu-1+L2x0;  //epsilon = -1 -> L1
  std::cout << "L1: " << L1_x << ", L2: " << L2_x << std::endl;

  Vector L2 = {L2_x, 0.0, 0.0};
  Vector L1 = {L1_x, 0.0, 0.0};
  auto eigenvals_L2 = fixpoint_stability(L2, mu); //vector<EigenPairs>
  auto eigenvals_L1 = fixpoint_stability(L1, mu);
  
  std::cout << "L2 fixpoint eigenvals\n";
  for (const auto& ev : eigenvals_L2){
    std::cout << "Re: " << ev.real << ", Im: " << ev.imag << std::endl;
  };

  std::cout << "L1 fixpoint eigenvals\n";
  for (const auto& ev : eigenvals_L1){
    std::cout << "Re: " << ev.real << ", Im: " << ev.imag << std::endl;
  };

  //pass mu to CR3BP
  auto f = [mu](double t, const Vector& y) -> Vector {
    return crtbp_with_STM(t, y, mu);
  };

  //initial guess
  Vector y0 = {-1.011035058, 0.0, -0.1731500000, 0.0, 0.078014119, 0.0};

  //extend with identity STM
  Vector STM_flat(36, 0.0);
  for (int i = 0; i < 6; i++)
    STM_flat[i*6 + i] = 1.0;
  Vector state_ext = y0;
  state_ext.insert(state_ext.end(), STM_flat.begin(), STM_flat.end());
    
  //initial step size
  double h = 1e-4; //adaptive step size will handle it if it's too large
  
  //some tests of important functions

  auto no_event = [](double t, const Vector& y){ return 1.0; };
  SolverResult res = solve_de(0.0, 5, state_ext, f, no_event, false, h, 1e-10, 1e-8);
  for (size_t i = 0; i < res.solution.rows(); i += 10) {
      std::cout << "t=" << res.times[i] 
                << " x=" << res.solution(i,0)
                << " y=" << res.solution(i,1)
                << " z=" << res.solution(i,2) << "\n";
  }
  

  //integrate to xz test
  std::cout << "\nTEST integrate_to_xz\n";
  try {
    auto res = integrate_to_xz(state_ext, 5.0, h, f, 1e-10, 1e-8);
    std::cout << "Half period: " << res.half_time << "\n";
    std::cout << "State at T/2: [";
    for (double v : res.half_state) std::cout << v << " ";
    std::cout << "]\n";
  } catch (const std::exception& e) {
    std::cerr << "integrate_to_xz failed: " << e.what() << "\n";
  }

  //halo differential correction test
  std::cout << "\nTEST halo_differential_correction\n";
  DiffCorrResult corr;
  try {
    corr = halo_differential_correction(state_ext, 5.0, h, f, 1e-8, 1e-10, 1e-8);
    std::cout << "Corrected period: " << corr.period << "\n";
    std::cout << "Corrected state: [";
    for (double v : corr.corrected_state) std::cout << v << " ";
    std::cout << "]\n";

    //quick check: vx and vz at T/2 should be ~0
    auto check = integrate_to_xz(corr.corrected_state, corr.period, h, f, 1e-10, 1e-8);
    std::cout << "vx at T/2 (should be ~0): " << check.half_state[3] << "\n";
    std::cout << "vz at T/2 (should be ~0): " << check.half_state[5] << "\n";
  } catch (const std::exception& e) {
      std::cerr << "halo_differential_correction failed: " << e.what() << "\n";
  }

  std::cout << "\nTEST pseudo_arclength\n";
  ContinuationResult cont_neg;
  try {
    cont_neg = pseudo_arclength(f, corr.corrected_state,
                                                corr.period, mu,
                                                -1e-3, h, 200,
                                                1e-8, 1e-10, 1e-8);
    std::cout << "Number of orbits: " << cont_neg.periods.size() << "\n";
    for (size_t i = 0; i < cont_neg.periods.size(); i++) {
        std::cout << "T[" << i << "] = " << cont_neg.periods[i]
                  << "  x0 = " << cont_neg.states(i, 0)
                  << "  y0 = " << cont_neg.states(i, 1)
                  << "  z0 = " << cont_neg.states(i, 2)
                  << "  vx0 = " << cont_neg.states(i, 3)
                  << "  vy0 = " << cont_neg.states(i, 4)
                  << "  vz0 = " << cont_neg.states(i, 5) << "\n";
    }
  } catch (const std::exception& e) {
      std::cerr << "pseudo_arclength failed: " << e.what() << "\n";
  }

  std::cout << "\nTEST pseudo_arclength other direction\n";
  ContinuationResult cont_pos;
  try {
    cont_pos = pseudo_arclength(f, corr.corrected_state,
                                                corr.period, mu,
                                                1e-3, h, 100,
                                                1e-8, 1e-10, 1e-8);
    std::cout << "Number of orbits: " << cont_pos.periods.size() << "\n";
    for (size_t i = 0; i < cont_pos.periods.size(); i++) {
        std::cout << "T[" << i << "] = " << cont_pos.periods[i]
                  << "  x0 = " << cont_pos.states(i, 0)
                  << "  z0 = " << cont_pos.states(i, 2)
                  << "  vy0 = " << cont_pos.states(i, 4) << "\n";
    }
  } catch (const std::exception& e) {
      std::cerr << "pseudo_arclength failed: " << e.what() << "\n";
  }

  ContinuationResult full_family;
  full_family.states = cont_neg.states;
  full_family.periods = cont_neg.periods;
  //append cont_pos skipping index 0 (duplicate initial orbit)
  for(size_t i = 1; i < cont_pos.periods.size(); i++) {
    full_family.states.resize(full_family.periods.size()+1, 6);
    full_family.states.setRow(full_family.periods.size(), cont_pos.states.row(i));
    full_family.periods.push_back(cont_pos.periods[i]);
  }

  std::vector<SolverResult> trajectories;
  std::vector<std::pair<double, double>> all_nu;
  std::vector<double> all_lambda;

  for (size_t i=0; i<full_family.periods.size(); i++){
    Vector y0 = extend_with_STM(full_family.states, i);
    auto no_event = [](double t, const Vector& y){return 1.0;};
    SolverResult res = solve_de(0.0, full_family.periods[i], y0, f, no_event, false, h, 1e-10, 1e-8);
    trajectories.push_back(res);

    //stability indices
    size_t last = res.solution.rows()-1;
    all_nu.push_back(stability_indices(res, last));

    //Lyapunov expos using 3 periods
    SolverResult res3 = solve_de(0.0, full_family.periods[i]*3.0, y0, f, no_event, false, h, 1e-10, 1e-8, true);
    all_lambda.push_back(res3.cumulative_log[0]/(3.0*full_family.periods[i]));
  }

  write_trajectories(trajectories, "./nrhos/trajectories.csv");
  write_family(full_family, all_nu, all_lambda, "./nrhos/family.csv", mu, L1_x, L2_x);

  return 0;
}
