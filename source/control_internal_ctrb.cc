// #include <unsupported/Eigen/MatrixFunctions>
#include "control_internal.h"

namespace control {
namespace internal {

template <typename Derived>
Derived ctrb(const Eigen::MatrixBase<Derived>& A,
             const Eigen::MatrixBase<Derived>& B)
{
  const auto n = A.rows();
  const auto m = B.cols();

  Derived result(n, n*m);
  Derived A_to_the_i = Derived::Identity(n, n);
  result.block(0, 0, n, m) = B;
  for (int i = 1; i < n; ++i) {
    A_to_the_i *= A;
    result.block(0, i*m, n, m) = A_to_the_i * B;
  }
//  for (int i = 1; i < n; ++i)
//    result.block(0, i*m, n, m) = A.pow(i).eval() * B;

  return result;
}

// Explicit template instantiation for double
template Eigen::MatrixXd ctrb<Eigen::MatrixXd>(
    const Eigen::MatrixBase<Eigen::MatrixXd>& A,
    const Eigen::MatrixBase<Eigen::MatrixXd>& B);

} // namespace internal
} // namespace control


