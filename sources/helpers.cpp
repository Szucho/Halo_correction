#include <cmath>
#include "../headers/matrix.h"
#include <vector>


/*
 helpers.cpp  MINPACK QR factorization and Gauss elimination
 Bertalan Szuchovszky 12.03.2026

 Implementation of functions declared in helpers.h.

 qform and qrfac are translated from the MINPACK Fortran library
 (Argonne National Laboratory, 1980) into C++ using the Vector/Matrix
 types from matrix.h. The algorithms are otherwise unchanged
 Variable names and loop structure follow the original source closely so that
 the MINPACK documentation and comments remain applicable.

 qrfac: QR factorization with optional column pivoting via Householder
        reflections. Overwrites the input matrix in-place with the
        Householder vectors; diagonal of R is stored in rdiag. -MINPACK

 qform: Accumulates the Householder reflections from qrfac into the
        explicit orthogonal factor Q. Must be called after qrfac if
        Q is needed explicitly (e.g. for STM re-orthogonalization in
        the Lyapunov exponent computation in solve_de). -MINPACK

 norm_v: Euclidean norm of a subvector v[start : start+len].
         Pulled out as a helper since both qrfac and qform need it
         repeatedly and std::norm operates on complex numbers.

 gauss_elimination: standard Gaussian elimination with partial pivoting
                    on an augmented matrix [A | b] of size dim x (dim+1).
                    Returns the solution vector x of Ax = b.
                    Used in newton_gauss_2x2 and newton_gauss_3x3 in
                    diff_correction.h for the halo orbit corrector.

 Dependencies: matrix.h
*/


using namespace VecOps;

//MINPACK QR-factorization
void qform(int m, int n, Vector&q, int ldq, Vector&wa){

  if ((int)q.size() < m * ldq) throw std::invalid_argument("q too small");
  if ((int)wa.size() < m)      throw std::invalid_argument("wa too small");

  int i, j, k, l, minmn, np1;
  double sum, temp;

  minmn = std::min(m, n);

  if (minmn >= 2){
    for (j=0; j<minmn; j++){
      for (i =0; i<j; i++){
        q[i + j * ldq] = 0.0;
      }
    }
  }

  np1 = n+1;
  if (m >= np1){
    for (j = np1-1; j<m; j++){
      for (i=0; i<m; i++){
        q[i + j*ldq] = 0.0;
      }
    q[j + j*ldq] = 1.0;
    }
  }


  for (l=0; l<minmn; l++){
    k = minmn-l-1;
    for (i = k; i<m; i++){
      wa[i] = q[i + k*ldq];
      q[i + k*ldq] = 0.0;
    }
    q[k + k*ldq] = 1.0;
    if(wa[k]==0){continue;}
    for(j=k; j<m;j++){
      sum = 0.0;
      for (i=k; i<m; i++){
        sum +=q[i + j*ldq]*wa[i];
      }
      temp = sum/wa[k];
      for (i=k; i<m; i++){
        q[i +j*ldq] -= temp*wa[i];
      }
    }
  }
  return;
}

double norm_v(const Vector& v, int start, int len){
  double sum = 0.0;
  for (int i=start; i<start+len; i++){
      sum += v[i]*v[i];
  }
  return std::sqrt(sum);
}

void qrfac(int m, int n, Vector& a, int lda,
           bool pivot, std::vector<int>& ipvt, int lipvt, 
           Vector& rdiag, Vector& acnorm, Vector& wa) {

  if ((int)a.size() < m * lda)       throw std::invalid_argument("a too small");
  if ((int)rdiag.size() < n)         throw std::invalid_argument("rdiag too small");
  if ((int)acnorm.size() < n)        throw std::invalid_argument("acnorm too small");
  if ((int)wa.size() < n)            throw std::invalid_argument("wa too small");
  if (pivot && (int)ipvt.size() < n) throw std::invalid_argument("ipvt too small");

  int i, j, jp1, k, kmax, minmn;
  double ajnorm, sum, temp, epsmch;

  epsmch = std::numeric_limits<double>::epsilon();

  for(j=0; j<n; j++){
    acnorm[j] = norm_v(a, j*lda, m);
    rdiag[j] = acnorm[j];
    wa[j] = rdiag[j];
    if(pivot) ipvt[j] = j;
  }
  
  minmn = std::min(m, n);
  for (j = 0; j<minmn; j++){
    if(pivot){
      kmax = j;
      for (k=j; k<n; k++){
        if(rdiag[k] > rdiag[kmax]){
          kmax = k;
        }
      }
      if(kmax != j){
        for (i=0; i<m; i++){
          temp = a[i + j*lda];
          a[i + j*lda] = a[i + kmax*lda];
          a[i + kmax*lda] = temp;
        }
        rdiag[kmax] = rdiag[j];
        wa[kmax] = wa[j];
        k = ipvt[j];
        ipvt[j] = ipvt[kmax];
        ipvt[kmax] = k;
      }
    }

    ajnorm = norm_v(a, j+j*lda, m-j);
    if(ajnorm == 0.0){continue;}
    for (i = j; i<m; i++){
      a[i +j*lda] /= ajnorm;
    }
    a[j + j*lda] += 1.0;

    jp1 = j + 1;
    if(n>=jp1){
      for (k = jp1; k<n; k++){
        sum = 0.0;
        for (i = j; i<m; i++){
          sum += a[i + j*lda]*a[i + k*lda];
        }
        temp = sum/a[j + j*lda];
        for (i=j; i<m; i++){
          a[i + k*lda] -= temp*a[i + j*lda];
        }
        if(pivot && rdiag[k] != 0.0){
          temp = a[j + k*lda] / rdiag[k];
          rdiag[k] *= std::sqrt(std::max(0.0, (1.0-temp*temp)));
        }
        if(0.05*(rdiag[k]/wa[k])*(rdiag[k]/wa[k]) <= epsmch){
          rdiag[k] = norm_v(a, jp1+k*lda, m-j);
          wa[k] = rdiag[k];
        }
      }
      rdiag[j] = -ajnorm;
    }
  }
  return;
}



//Gauss elimination with partial pivoting
//solves Ax = b where mat is the augmented matrix [A | b] (dim x dim+1) with b stored in it
Vector gauss_elimination(Matrix mat, int dim) {
  for (int k = 0; k < dim; k++) {
    //find max abs value in column k for partial pivoting
    int maxRow = k;
    for (int i = k + 1; i < dim; i++) {
      if (std::abs(mat(i, k)) > std::abs(mat(maxRow, k)))
        maxRow = i;
    }
    //swap rows
    if (maxRow != k) {
      for (int j = 0; j <= dim; j++) {
        std::swap(mat(k, j), mat(maxRow, j));
      }
    }
    //ALWAYS check for singularity before elimination
    double pivot = mat(k, k);
    if (std::abs(pivot) < 1e-14) //small floating point determinant errors may arise so exclude those too
      throw std::runtime_error("Gauss elimination: singular or nearly singular matrix");

    //normalize pivot row
    for (int j = k; j <= dim; j++)
      mat(k, j) /= pivot;

    //eliminate rows below pivot
    for (int i = k + 1; i < dim; i++) {
      double factor = mat(i, k);
      for (int j = k; j <= dim; j++)
        mat(i, j) -= factor * mat(k, j);
    }
  }

  //back substitution
  Vector result(dim, 0.0);
  for (int i = dim - 1; i >= 0; i--) {
    result[i] = mat(i, dim);
    for (int j = i + 1; j < dim; j++)
      result[i] -= mat(i, j) * result[j];
  }
  return result;
}
