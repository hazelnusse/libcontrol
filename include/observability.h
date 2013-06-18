#ifndef OBSERVABILITY_H
#define OBSERVABILITY_H

#include <Eigen/Dense>

namespace control {

template <typename Derived>
Derived observability_matrix(const Eigen::MatrixBase<Derived>& A,
                             const Eigen::MatrixBase<Derived>& C);

template <typename Derived>
bool is_observable(const Eigen::MatrixBase<Derived>& A,
                   const Eigen::MatrixBase<Derived>& C);

}

#endif // OBSERVABILITY_H

