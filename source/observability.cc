#include <stdexcept>
#include "observability.h"
#include "control_internal.h"

namespace control {

template <typename Derived>
Derived observability_matrix(const Eigen::MatrixBase<Derived>& A,
              const Eigen::MatrixBase<Derived>& C)
{
  if (A.rows() != A.cols())
    throw std::invalid_argument("A must be square.");

  if (A.rows() != C.cols())
    throw std::invalid_argument("A and C must have the same number of columns.");

  const Derived At = A.transpose();
  const Derived Ct = C.transpose();
  Derived res = internal::ctrb(At, Ct).transpose();
  return res;
}

template <typename Derived>
bool is_observable(const Eigen::MatrixBase<Derived>& A,
                   const Eigen::MatrixBase<Derived>& C)
{
  Derived O = control::observability_matrix(A, C);
  Eigen::FullPivHouseholderQR<Derived> decomposition(O);

  if (decomposition.rank() != O.cols())
    return false;

  return true;
}

// Explicit template instantiation
template Eigen::MatrixXd observability_matrix<Eigen::MatrixXd>(
    const Eigen::MatrixBase<Eigen::MatrixXd>& A,
    const Eigen::MatrixBase<Eigen::MatrixXd>& C);

template bool is_observable<Eigen::MatrixXd>(
    const Eigen::MatrixBase<Eigen::MatrixXd>& A,
    const Eigen::MatrixBase<Eigen::MatrixXd>& C);

} // namespace control

