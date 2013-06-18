#ifndef DARE_H
#define DARE_H

#include <Eigen/Dense>

namespace control {

/** \brief Solve discrete-time algebraic Riccati equation.
 *
 * Solve the following discrete-time algebraic Riccati equation:
 *
 *  F ^ T * X * F - X - F ^ T * X * G1 * (G2 + G1 ^ T * X * G1) ^ -1 * G1 ^ T * X * F + H = 0
 *
 * If you are familiar to using Matlab's dare() function, you should use the
 * following arguments:
 *
 * F = A
 * G1 = B
 * G2 = R
 * H = Q
 *
 * where A, B, R, and Q are the arguments required by Matlab's dare() function,
 * typically dare(A, B, Q, R)
 *
 * */
template <typename Derived>
Derived dare(const Eigen::MatrixBase<Derived>& F,
             const Eigen::MatrixBase<Derived>& G1,
             const Eigen::MatrixBase<Derived>& G2,
             const Eigen::MatrixBase<Derived>& H);

}

#endif // DARE_H

