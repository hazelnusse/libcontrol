#include <stdexcept>
#include "controllability.h"
#include "control_internal.h"

namespace control {

template <typename Derived>
Derived controllability_matrix(const Eigen::MatrixBase<Derived>& A,
              const Eigen::MatrixBase<Derived>& B)
{
  if (A.rows() != A.cols())
    throw std::invalid_argument("A must be square.");

  if (A.rows() != B.rows())
    throw std::invalid_argument("A and B must have the same number of rows.");

  return control::internal::ctrb(A, B);
}

template <typename Derived>
bool is_controllable(const Eigen::MatrixBase<Derived>& A,
                     const Eigen::MatrixBase<Derived>& B)
{
  Derived C = control::internal::ctrb(A, B);
  Eigen::FullPivHouseholderQR<Derived> decomposition(C);

  if (decomposition.rank() != C.rows())
    return false;

  return true;
}

// Explicit template instantiation
template Eigen::MatrixXd controllability_matrix<Eigen::MatrixXd>(
    const Eigen::MatrixBase<Eigen::MatrixXd>& A,
    const Eigen::MatrixBase<Eigen::MatrixXd>& B);

template bool is_controllable<Eigen::MatrixXd>(
    const Eigen::MatrixBase<Eigen::MatrixXd>& A,
    const Eigen::MatrixBase<Eigen::MatrixXd>& B);

} // namespace control

