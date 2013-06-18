#ifndef CARE_H
#define CARE_H

#include <Eigen/Dense>

namespace control {

/** \brief Solve continuous-time algebraic Riccati equation.
 *
 * Solve the following continuous-time algebraic Riccati equation:
 *
 *  F ^ T * X + X * F - X * G * X + H = 0
 *
 * If you are familiar to using Matlab's care() function, you should use the
 * following arguments:
 *
 * F = A
 * G = B * R^-1 * B^T
 * H = Q
 *
 * where A, B, and Q are the arguments required by Matlab's care() function,
 * typically care(A, B, Q, R)
 *
 * */
template <typename Derived>
Derived care(const Eigen::MatrixBase<Derived>& F,
             const Eigen::MatrixBase<Derived>& G,
             const Eigen::MatrixBase<Derived>& H);

}

#endif // CARE_H

