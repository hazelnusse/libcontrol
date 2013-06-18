#include <iostream>
#include <Eigen/Dense>
#include "gtest/gtest.h"
#include "observability.h"

TEST(ObservabilityTest, Simple)
{
  Eigen::MatrixXd A(2, 2);
  Eigen::MatrixXd C(2, 2);
  A << 1, 1, 4, -2;
  C << 1, 0, 0, 1;
  Eigen::MatrixXd O = control::observability_matrix(A, C);
  Eigen::MatrixXd O_expected(4, 2);
  O_expected << 1, 0,
                0, 1,
                1, 1,
                4, -2; // From Matlab.
  EXPECT_EQ(O, O_expected) << "Observability matrix differs: " << O_expected
      << std::endl << O << std::endl;
}

