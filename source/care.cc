#include <complex>
#define HAVE_LAPACK_CONFIG_H
#define LAPACK_COMPLEX_CPP
#include <lapacke.h>
#include <stdexcept>
#include <string>
#include <unsupported/Eigen/MatrixFunctions>

#include "care.h"
#include "controllability.h"
#include "observability.h"

namespace control {

namespace internal {
  lapack_logical select_lhp(const double *real, const double *imag)
  {
    return *real < 0.0;
  }
}

template <typename Derived>
Derived care(const Eigen::MatrixBase<Derived>& F,
             const Eigen::MatrixBase<Derived>& G,
             const Eigen::MatrixBase<Derived>& H)
{
  if (F.rows() != F.cols())
    throw std::invalid_argument("F must be square.");

  if (G.rows() != G.cols())
    throw std::invalid_argument("G must be square.");
  
  if (H.rows() != H.cols())
    throw std::invalid_argument("H must be square.");
  
  if (G.rows() != F.rows())
    throw std::invalid_argument("F and G must be of same dimensions.");

  if (!G.isApprox(G.transpose().eval()))
    throw std::invalid_argument("G must be symmetric.");

  if (H.rows() != F.rows())
    throw std::invalid_argument("F and H must be of same dimensions.");

  if (!H.isApprox(H.transpose().eval()))
    throw std::invalid_argument("H must be symmetric.");
  
  Eigen::FullPivHouseholderQR<Derived> dec_G(G);
  if (!dec_G.isInvertible())
    throw std::invalid_argument("G must be non-singular.");
  
  Eigen::FullPivHouseholderQR<Derived> dec_H(H);
  if (!dec_H.isInvertible())
    throw std::invalid_argument("H must be non-singular.");

  const Derived G_sqrt = G.sqrt();
  if (!control::is_controllable(F, G_sqrt))     // actually, only stabilizibility is required
    throw std::invalid_argument("The pair (F, G^(1/2)) must be controllable.");
  
  const Derived H_sqrt = H.sqrt();
  if (!control::is_observable(H_sqrt, F))       // actually, only detectability is required
    throw std::invalid_argument("The pair (H^(1/2), F) must be observable.");

  const auto n = F.rows();
  Derived Z(2*n, 2*n);          // Hamiltonian matrix
  Z.block(0, 0, n, n) = F;
  Z.block(0, n, n, n) = -G;
  Z.block(n, 0, n, n) = -H;
  Z.block(n, n, n, n) = -F.transpose();

  // form ordered Schur decomposition of Z
  Derived U(2*n, 2*n);          // Orthogonal matrix from Schur decomposition
  Eigen::VectorXd WR(2*n);
  Eigen::VectorXd WI(2*n);
  lapack_int sdim = 0;                 // Number of eigenvalues for which sort is true
  lapack_int info;
  info = LAPACKE_dgees(LAPACK_COL_MAJOR,    // Eigen default storage order
                       'V',                 // Schur vectors are computed
                       'S',                 // Eigenvalues are sorted
                       internal::select_lhp,// Ordering callback
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
  
  return U11.fullPivHouseholderQr().solve(U21).transpose();
}

// Explicit template instantiations for double
template Eigen::MatrixXd care<Eigen::MatrixXd>(
    const Eigen::MatrixBase<Eigen::MatrixXd>& F,
    const Eigen::MatrixBase<Eigen::MatrixXd>& G,
    const Eigen::MatrixBase<Eigen::MatrixXd>& H);

} // namespace control


