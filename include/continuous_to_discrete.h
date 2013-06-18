#ifndef CONTINUOUS_TO_DISCRETE_H
#define CONTINUOUS_TO_DISCRETE_H

#include <Eigen/Dense>
#include <utility>

namespace control {

template <typename Derived>
std::pair<Derived, Derived> continuous_to_discrete(const Eigen::MatrixBase<Derived>& A,
                               const Eigen::MatrixBase<Derived>& B,
                               double Ts);

}

#endif // CONTINUOUS_TO_DISCRETE_H

