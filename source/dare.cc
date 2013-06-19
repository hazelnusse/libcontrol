#include <complex>
#define HAVE_LAPACK_CONFIG_H
#define LAPACK_COMPLEX_CPP
#include <lapacke.h>
#include <stdexcept>
#include <string>
// #include <unsupported/Eigen/MatrixFunctions>

#include "controllability.h"
#include "dare.h"
#include "observability.h"

namespace control {

namespace internal {
  lapack_logical select_iuc(const double *real, const double *imag)
  {
    return std::norm(std::complex<double>(*real, *imag)) < 1.0;
  }
}

template <typename Derived>
Derived dare(const Eigen::MatrixBase<Derived>& F,
             const Eigen::MatrixBase<Derived>& G1,
             const Eigen::MatrixBase<Derived>& G2,
             const Eigen::MatrixBase<Derived>& H)
{
  if (F.rows() != F.cols())
    throw std::invalid_argument("F must be square.");
  
  if (H.rows() != H.cols())
    throw std::invalid_argument("H must be square.");

  if (G2.rows() != G2.cols())
    throw std::invalid_argument("G2 must be square.");
  
  if (H.rows() != F.rows())
    throw std::invalid_argument("F and H must be of same dimensions.");

  if (G1.rows() != F.rows())
    throw std::invalid_argument("G1 and F must have the same number of rows.");

  if (G1.cols() != G2.cols())
    throw std::invalid_argument("G1 and G2 must have the same number of columns.");

  if (G2.rows() > F.rows())   // possible typo in Laub 1979, he writes m < n, not m <= n
    throw std::invalid_argument("G2 must not have more rows than F.");

  if (!H.isApprox(H.transpose().eval()))
    throw std::invalid_argument("H must be symmetric.");
  const Derived H_sym = (H + H.transpose()) / 2;    // Force exact symmetry
  
  if (!G2.isApprox(G2.transpose().eval()))
    throw std::invalid_argument("G2 must be symmetric.");
  const Derived G2_sym = (G2 + G2.transpose()) / 2;   // Force exact symmetry

  Eigen::LDLT<Derived> ldlt_H(H_sym);
  if ((ldlt_H.info() != Eigen::Success) || ldlt_H.isNegative())
    throw std::invalid_argument("H must positive semi-definite.");
  
  Eigen::FullPivHouseholderQR<Derived> dec_G2(G2_sym);
  if (!dec_G2.isInvertible())
    throw std::invalid_argument("G2 must be non-singular.");

  Eigen::FullPivHouseholderQR<Derived> dec_F(F);
  if (!dec_F.isInvertible())
    throw std::invalid_argument("F must be non-singular.");

  if (!control::is_controllable(F, G1)) // actually, only stabilizibility is required
    throw std::invalid_argument("The pair (F, G1) must be controllable.");
  
  // TODO: Add these checks once matrix power / square root code stabilizes in
  // Eigen
//  const Derived H_sqrt = H.pow(0.5);
//  if (!control::is_observable(H_sqrt, F)) // actually, only detectability is required
//    throw std::invalid_argument("The pair (H^(1/2), F) must be observable.");

  const Derived G = G1 * dec_G2.solve(G1.transpose());
  const Derived Finv_t = dec_F.inverse().transpose();
  const auto n = F.rows();
  Derived Z(2*n, 2*n);
  Z.block(0, 0, n, n) = F + G * Finv_t * H_sym;
  Z.block(0, n, n, n) = -G * Finv_t;
  Z.block(n, 0, n, n) = -Finv_t * H_sym;
  Z.block(n, n, n, n) = Finv_t;
  Derived U(2*n, 2*n);

  // form ordered Schur decomposition of Z
  Eigen::VectorXd WR(2*n);
  Eigen::VectorXd WI(2*n);
  lapack_int sdim = 0;                 // Number of eigenvalues for which sort is true
  lapack_int info;
  info = LAPACKE_dgees(LAPACK_COL_MAJOR,    // Eigen default storage order
                       'V',                 // Schur vectors are computed
                       'S',                 // Eigenvalues are sorted
                       internal::select_iuc,// Ordering callback
                       Z.rows(),            // Dimension of test matrix
                       Z.data(),            // Pointer to first element
                       Z.rows(),            // Leading dimension (column stride)
                       &sdim,               // Number of eigenvalues sort is true
                       WR.data(),           // Real portion of eigenvalues
                       WI.data(),           // Complex portion of eigenvalues
                       U.data(),            // Orthogonal transformation matrix
                       Z.rows());           // Dimension of Z

  if (info < 0) {
    std::string err = "The " + std::to_string(info) +
                      "-th argument to LAPACK's dgees() had an illegal value.";
    throw std::invalid_argument(err);
  } else if (info > 0) {
    std::string err = "LAPACK's dgees() function was not successful, return error was " +
                      std::to_string(info) + ".";
    throw std::invalid_argument(err);
  }

  if (sdim != n) {
    std::string err = "Symplectic matrix only has " + std::to_string(sdim)
               + " eigenvalues inside unit circle, when it should have "
               + std::to_string(n);
    throw std::invalid_argument(err);
  }

  Derived U11 = U.block(0, 0, n, n).transpose();
  Derived U21 = U.block(n, 0, n, n).transpose();
  
  return U11.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(U21).transpose();
}

// Explicit template instantiations for double
template Eigen::MatrixXd dare<Eigen::MatrixXd>(
    const Eigen::MatrixBase<Eigen::MatrixXd>& F,
    const Eigen::MatrixBase<Eigen::MatrixXd>& G1,
    const Eigen::MatrixBase<Eigen::MatrixXd>& G2,
    const Eigen::MatrixBase<Eigen::MatrixXd>& H);

} // namespace control

