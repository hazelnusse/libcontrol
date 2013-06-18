#include <cmath>
#include <stdexcept>
#include "continuous_to_discrete.h"

namespace control {

template <typename Derived>
std::pair<Derived, Derived> continuous_to_discrete(const Eigen::MatrixBase<Derived>& A,
                                  const Eigen::MatrixBase<Derived>& B,
                                  double Ts)
{
  if (A.rows() != A.cols())
    throw std::invalid_argument("A must be square.");

  if (A.rows() != B.rows())
    throw std::invalid_argument("A and B must have same number of rows.");

  const auto n = A.rows();

  // Eigen lacks a reliable matrix exponential function, and LAPACK apparently
  // does not have one either.  While there are more sophisticated ways to
  // compute the matrix exponential, the Taylor series approach is the simplest
  // to implement.  It can fail to converge for certain input matrices though,
  // so beware....
  Derived eAT = Derived::Identity(n, n);
  Derived eAT_prev = Derived::Zero(n, n);
  Derived An_prev = A;
  uint64_t i = 1;
  uint64_t factorial = 1;
  while (!eAT.isApprox(eAT_prev)) {
    eAT_prev = eAT;      // Save the current approximation of eAT
    eAT += (std::pow(Ts, i) / factorial) * An_prev;
    ++i;
    factorial *= i;
    An_prev *= A;
  }

  Eigen::FullPivHouseholderQR<Derived> A_qr(A);
  Derived B_discrete = A_qr.solve(eAT - Derived::Identity(n, n)) * B;

  return std::make_pair(eAT, B_discrete);
}

// Explicit template instantiation for double
template std::pair<Eigen::MatrixXd, Eigen::MatrixXd> continuous_to_discrete<Eigen::MatrixXd>(
    const Eigen::MatrixBase<Eigen::MatrixXd>& A,
    const Eigen::MatrixBase<Eigen::MatrixXd>& B,
    double Ts);

} // namespace control



