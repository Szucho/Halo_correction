#ifndef SOLVER_H
#define SOLVER_H

#include "matrix.h"
#include "helpers.h"
#include <cmath>

/*
 solver.h ODE solver and root finding utilities
 Bertalan Szuchovszky 12.03.2026

 Numerical methods for solving ODEs and finding roots, using matrix.h
 for numpy-like array (here vector in c++) operations.
 Yes it's in a header file because I use templates.
 If you have a better solution message me!

 Contents:
  - brent_method:    Brent root finding (bisection/halving + secant + IQI)
                     guaranteed convergence if f(a)*f(b) < 0

  - dorpri54:        single step of Dormand-Prince 5(4) method
                     returns both 5th order solution (y1) and
                     4th order solution (yhat) for error estimation
                     Butcher tableau: Dormand & Prince (1980)

  - solve_de:        adaptive ODE solver built on dorpri54 (Dormand-Prince 54)
                     - step size control via atol/rtol error norm (shifting error)
                     - event detection with Brent root finding
                       (linear interpolation instead of re-integration
                        for performance => see comments about re-integration in code)
                     - optional Lyapunov exponent computation via
                       QR factorization of the State Transition Matrix
                       at each accepted step (see helpers.h for QR)
                       method: Dieci et al. (1997), DOI: 10.1016/S0167-2789(96)00216-3
                       ONLY for STM enhanced systems where dPhi is flattened
                       and contained in the derivative of the state vector
                       IF STM is singular (det=0) the solver might ropemax & hang itself

 Usage:
   !!!DON'T FORGET THIS!!! very important to define a no event if we don't want to check for events
   auto no_event = [](double t, const Vector& y){ return 1.0; };
   SolverResult res = solve_de(t0, tf, y0, f, no_event, false, h0, atol, rtol); lyapunov is false by default
   // res.solution       -> Matrix, rows = timesteps, cols = state vars
   // res.event_times    -> Vector, empty if no events fired, contains the time of the events
   // res.cumulative_log -> Vector, empty if lyapunov=false, accumulated log|R_{ii}| for Lyapunov exponents
                       
 Note: for Lyapunov computation y0 must be [physical state, STM flattened]
       sys_dim is derived automatically from y0.size()
       if this hangs itself it's a skill issue, user might want to check if y0 has correct dimension
       other option would have been an int sys_dim function input of solve_de but we are better than that
*/

struct DOPRI54Result {
    Vector y1;   //5th order solution
    Vector yhat; //4th order solution
};


struct SolverResult{
  Matrix solution;       //trajectory: rows = timesteps, cols = state vars
  Vector times;          //timesteps t, remember that the integrator uses adaptive dt
  Vector event_times;    //times where event happened
  Matrix event_states;   //state vector at event time
  Vector cumulative_log; //accumulated log|R_{ii}| for Lyapunov exponents
};


//brent method
template<typename FuncType>
double brent_method(FuncType f, double a, double b, double tolerance = 1e-12) {
  double fa = f(a), fb = f(b);
  if (fa * fb > 0)
    throw std::runtime_error("f(a) and f(b) must have opposite signs.");

  //let |f(a)| >= |f(b)|, if not -> swap(a, b)
  if (std::abs(fa) < std::abs(fb)) {
    std::swap(a, b);
    std::swap(fa, fb);
  }

  double c = a, fc = fa;
  bool mflag = true;  //flag to distinguish halving and interpolation steps
  double s = b;
  double d = 0.0;     //step size
  int max_iter=1000;
  int iter=0;

  while (std::abs(fb) > tolerance && std::abs(b - a) > tolerance) {
    if (iter++>max_iter) throw std::runtime_error("Brent method max iter reached");
    if (fa != fc && fb != fc) {
        //inverse quadratic interpolation
        s = a * fb * fc / ((fa - fb) * (fa - fc))
          + b * fa * fc / ((fb - fa) * (fb - fc))
          + c * fa * fb / ((fc - fa) * (fc - fb));
    } else {
        //secant method
        s = b - fb * (b - a) / (fb - fa);
    }

    //conditions for accepting interpolation
    bool cond1 = (s - (3*a+b)/4) * (s - b) > 0;
    bool cond2 = (mflag && std::abs(s - b) >= std::abs(b - c) / 2);
    bool cond3 = (!mflag && std::abs(s - b) >= std::abs(c - d) / 2);
    bool cond4 = (mflag && std::abs(b - c) < tolerance);
    bool cond5 = (!mflag && std::abs(c - d) < tolerance);

    if (cond1 || cond2 || cond3 || cond4 || cond5) {
        //bisection/halving method
        s = (a + b) / 2;
        mflag = true;
    } else {
        mflag = false;
    }

    double fs = f(s);
    d = c;    //save c
    c = b;    //update c c=b
    fc = fb;  //update fc as fc=fb

    //f(a) & f(b) sign doesnt match
    if (fa * fs < 0) {
        b = s;
        fb = fs;
    } else {
        a = s;
        fa = fs;
    }

    //make sure that |f(a)| >= |f(b)|
    if (std::abs(fa) < std::abs(fb)) {
      std::swap(a, b);
      std::swap(fa, fb);
    }
  }

  return b;
}


//Dormand-Prince step, just check the Butcher tableau online 
template<typename FuncType>
DOPRI54Result dorpri54(double t, const Vector& y, FuncType f, double h){
  using namespace VecOps; //only needed for vec operations (+,-,*...) locally, otherwise it might break something :/
  //using the VecOps namespace allows me to use Vector as I would a numpy array
  //these are the vectorial k_{ni} values according to the Butcher tableau
  Vector k1 = f(t, y); //we assume this form for every f function containing the ODE
  Vector k2 = f(t+0.2*h, y+0.2*h*k1);
  Vector k3 = f(t+0.3*h, y+3.0/40.0*h*k1+9.0/40.0*h*k2);
  Vector k4 = f(t+0.8*h, y+44.0/45.0*h*k1-56.0/15.0*h*k2+32.0/9.0*h*k3);
  Vector k5 = f(t+8.0/9.0*h, y+19372.0/6561.0*h*k1-25360.0/2187.0*h*k2+64448.0/6561.0*h*k3-212.0/729.0*h*k4);
  Vector k6 = f(t+h, y+h*(9017.0/3168.0*k1-355.0/33.0*k2+46732.0/5247.0*k3+49.0/176.0*k4-5103.0/18656.0*k5));
  Vector k7 = f(t+h, y+h*(35.0/384.0*k1+500.0/1113.0*k3+125.0/192.0*k4-2187.0/6784.0*k5+11.0/84.0*k6));

  Vector y1=y+h*(35.0/384.0*k1+500.0/1113.0*k3+125.0/192.0*k4-2187.0/6784.0*k5+11.0/84.0*k6);
  Vector yhat=y+h*(5179.0/57600.0*k1+7571.0/16695.0*k3+393.0/640.0*k4-92097.0/339200.0*k5+187.0/2100.0*k6+1.0/40.0*k7);

  return {y1, yhat};
}



//adaptive solver - the creme de la creme, calculates trajectory, flags events, gets Lyapunov exponents
//gets you a BONUS @ your JOB buys you a CAR and helps you MOVE OUT OF THE HOUSE and helps you get a JET a YACHT...
template<typename FuncType, typename EventFuncType>
SolverResult solve_de(double t0, double tf, Vector y0, //t0,tf - initial and final  t, y0: initial condition (IVP)
                FuncType f, EventFuncType event,       //dot x = f(x,t), terminal event stops integration if event
                bool terminal, double h0, double atol, //h0:initial step size, absolute & relative tol for new h
                double rtol, bool lyapunov=false){     //if lyapunov calculates lyapunov expo-s, false by default

  using namespace VecOps;
  int sys_dim = (int)std::round((-1.0+std::sqrt(1.0+4.0*(double)y0.size()))/2.0);

  double t = t0;
  double h = h0;
  Vector y = y0;
  Vector times;
  times.push_back(t);
  double n_vars = (double)(sys_dim);
  Matrix result(1,y0.size()); //1 row at the beginning, new timesteps will be appended
  result.setRow(0,y);
  size_t step = 1; //will be needed to add new rows to the matrix since the
                   //solver is adaptive => we can't pre-allocate a fix sized matrix 
  
  //event setup
  Vector event_times;
  Matrix event_states;
  int event_num = 0;

  //Lyapunov setup
  std::vector<int> ipvt;
  Vector cumulative_log, rdiag, acnorm;
  if (lyapunov){
    ipvt.resize(sys_dim,0);
    rdiag.resize(sys_dim,1.0);
    acnorm.resize(sys_dim, 0.0);
    cumulative_log.resize(sys_dim,0.0); //resize as it is empty - no need to declare it if lyapunov=false
  }
  
  while(t<tf){
    h = std::min(h, tf-t);
    auto [y1, yhat] = dorpri54(t, y, f, h);

    
    double delt=0.0;
    for (int i=0; i<sys_dim; i++){
      double delta_i = std::abs(y1[i]-yhat[i]);
      double eps_i = atol + rtol*std::abs(y[i]);
      delt+=delta_i*delta_i/(eps_i*eps_i);
    }
    double delta = std::sqrt(1.0/n_vars*delt);
    if (delta<1.0){ //condition to accept solution
      //check for event if the step is accepted
      double event_val_n  = event(t,y);
      double event_val_n1 = event(t+h, y1);
      if (event_val_n*event_val_n1 < 0){
        auto event_function = [&](double t_guess) {
          double alpha = (t_guess - t) / h;          //alpha in [0,1]
          Vector y_interp = y + alpha * (y1 - y);    //linear interp - other option: calling dorpri but DON'T
          return event(t_guess, y_interp);           //calling dorpri will tank performance as brent will need
        };                                           //to take ~10-15 steps where we alway call the integ step
        double t_event = brent_method(event_function, t, t+h, 1e-14);
        //reusing interpolation -- other option: re-integration at t_event but that will slow down performance
        double alpha = (t_event-t)/h;
        Vector y_event = y + alpha*(y1-y); //lin interpol
        //storing event timte
        event_times.push_back(t_event);
        event_states.resize(event_num+1, y0.size());
        event_states.setRow(event_num, y_event);
        t = t_event; //stop/continue from event
        times.push_back(t); //save timestep
        y = y_event;
        result.resize(step+1, y0.size());
        result.setRow(step, y);
        step+=1;
        event_num+=1;
        if (terminal) break;
        else continue;
      }

      t+=h; //advance timestep if step is accepted
      times.push_back(t); //save current timestep
      y = y1; //store new accepted step (state vec)

      if (lyapunov) {
        //extract STM (last 36 components)
        Vector STM(y.begin() + sys_dim, y.begin() + sys_dim+sys_dim*sys_dim);
        Vector wa(sys_dim, 0.0);
        qrfac(sys_dim, sys_dim, STM, sys_dim, false, ipvt, sys_dim, rdiag, acnorm, wa);
        
        //accumulate log|R_ii|
        for (int j = 0; j < sys_dim; j++)
          cumulative_log[j] += std::log(std::abs(rdiag[j]));
        
        //re-orthogonalize: reset STM to Q via qform
        qform(sys_dim, sys_dim, STM, sys_dim, wa);
        
        //put Q back into state vector
        for (int i = 0; i < sys_dim*sys_dim; i++)
          y[sys_dim + i] = STM[i];
      }
      result.resize(step+1, y0.size()); //add one row
      result.setRow(step, y); //add solution to row
      step+=1; //increase number of steps
    }
    //adapt step size for the next step EVERY step (even if it is accepted)
    h = std::clamp(0.9 * h * std::pow(1.0/delta, 0.2), 0.1*h, 5.0*h); //dont let it grow too extreme

    if (h<1e-12){ //introducing minimum step size so that it won't be running forever
      throw std::runtime_error("Step size too small -> stiff problem");
    }
  }
  return {result, times, event_times, event_states, cumulative_log};
}

#endif // !SOLVER_H
