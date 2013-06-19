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

// From doi://10.1109/TAC.1979.1102178, Example 3
TEST(DARETest, Laub1979Example3)
{
  using Eigen::MatrixXd;
  MatrixXd A(2, 2);
  A << 0.9512, 0,
       0, 0.9048;
  MatrixXd B(2, 2);
  B << 4.877, 4.877,
      -1.1895, 3.569;
  MatrixXd R(2, 2);
  R << 1/3., 0,
          0, 3;
  MatrixXd Q(2, 2);
  Q << 0.005, 0,
       0, 0.02;

  MatrixXd X(2, 2);
  X = control::dare(A, B, R, Q);
  MatrixXd X_Laub(2, 2);
  X_Laub <<  0.010459082320970, 0.003224644477419,
             0.003224644477419, 0.050397741135643;

  EXPECT_TRUE(X.isApprox(X_Laub));
}
