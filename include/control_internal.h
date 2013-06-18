#ifndef CONTROL_INTERNAL_H
#define CONTROL_INTERNAL_H

#include <Eigen/Dense>

namespace control {
namespace internal {

template <typename Derived>
Derived ctrb(const Eigen::MatrixBase<Derived>& A,
             const Eigen::MatrixBase<Derived>& B);

}
}

#endif // CONTROL_INTERNAL_H

