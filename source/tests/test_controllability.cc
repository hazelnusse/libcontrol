#include <Eigen/Dense>
#include "gtest/gtest.h"
#include "controllability.h"

TEST(ControllabilityTest, Simple)
{
  Eigen::MatrixXd A(2, 2);
  Eigen::MatrixXd B(2, 2);
  A << 1, 1, 4, -2;
  B << 1, -1, 1, -1;
  Eigen::MatrixXd C = control::controllability_matrix(A, B);
  Eigen::MatrixXd C_expected(2, 4);
  C_expected << 1, -1, 2, -2, 
                1, -1, 2, -2; // From Matlab.
  EXPECT_EQ(C, C_expected);
}

