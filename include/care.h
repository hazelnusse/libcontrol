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
 * Where the equivalent Matlab call would be care(A, B, Q).
 *
 * which solves the Ricatti equation
 *
 * A ^ T * X + X * A - X * B * B^T * X + Q = 0
 *
 * */
template <typename Derived>
Derived care(const Eigen::MatrixBase<Derived>& F,
             const Eigen::MatrixBase<Derived>& G,
             const Eigen::MatrixBase<Derived>& H);

}

#endif // CARE_H

