#include <iostream>
#include <Eigen/Dense>
#include "gtest/gtest.h"

#include "care.h"

TEST(CARETest, Simple2d)
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
  MatrixXd G = B * R.inverse() * B.transpose();
  std::cout << G << std::endl;
  MatrixXd X(2, 2);
  X = control::care(A, G, Q);
  MatrixXd X_matlab(2, 2);
  X_matlab << 14.447642711356297, 13.925108341690663,
              13.925108341690663, 40.658332632601216;
  EXPECT_TRUE(X.isApprox(X_matlab));
}


