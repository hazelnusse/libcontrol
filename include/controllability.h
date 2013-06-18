#ifndef CONTROLLABILITY_H
#define CONTROLLABILITY_H

#include <Eigen/Dense>

namespace control {

template <typename Derived>
Derived controllability_matrix(const Eigen::MatrixBase<Derived>& A,
                               const Eigen::MatrixBase<Derived>& B);

template <typename Derived>
bool is_controllable(const Eigen::MatrixBase<Derived>& A,
                        const Eigen::MatrixBase<Derived>& B);

}

#endif // CONTROLLABILITY_H

