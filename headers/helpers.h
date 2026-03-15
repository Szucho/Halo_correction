#ifndef HELPERS_H
#define HELPERS_H

#include "matrix.h"

/*
 helpers.h MINPACK funcs translated to c++ by yours truly
 Bertalan Szuchovszky 12.03.2026
  
 I needed qfrom, qrfac in my solve_de function, norm_v was 
 needed in qrfac but it might be useful somewhere else too
 The source file is helpers.cpp, for explanations check MINPACK

 Gauss elimination is original but I mean it's a gauss elimination
 If You don't understand how to write a Gauss elimination
 function on your own I advise coming back to this project 
 later when You understand what's happening
*/

void qform(int m, int n, Vector&q, int ldq, Vector&wa);
double norm_v(const Vector& v, int start, int len);
void qrfac(int m, int n, Vector& a, int lda,
           bool pivot, std::vector<int>& ipvt, int lipvt, 
           Vector& rdiag, Vector& acnorm, Vector& wa);


//now this is my own creation but its just a simple forward elim backward sub standard Gauss method
Vector gauss_elimination(Matrix mat, int dim);

#endif // HELPERS_H
