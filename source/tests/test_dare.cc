#include <Eigen/Dense>
#include "gtest/gtest.h"

#include "dare.h"

TEST(DARETest, Simple2d)
{
  using Eigen::MatrixXd;
  MatrixXd A(2, 2);
  A << 0.814723686393179,   0.126986816293506, 0.905791937075619, 0.913375856139019;
  MatrixXd B(2, 2);
  B << 0.632359246225410, 0.278498218867048, 0.097540404999410, 0.546881519204984;
  MatrixXd Q(2, 2);
  Q << 1.1, 0, 0, 3.1;
  MatrixXd R(2, 2);
  R << 2.3, 0, 0, 68.25;
  MatrixXd X(2, 2);
  X = control::dare(A, B, R, Q);
  MatrixXd X_matlab(2, 2);
  X_matlab << 9.155517531254439, 5.432550918374424,
              5.432550918374424, 7.386819229053303;
  EXPECT_TRUE(X.isApprox(X_matlab));
}

