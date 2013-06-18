#include <Eigen/Dense>
#include <utility>
#include "gtest/gtest.h"
#include "continuous_to_discrete.h"

TEST(DiscretizeTest, Simple)
{
  Eigen::MatrixXd A(2, 2);
  Eigen::MatrixXd B(2, 2);
  A << 1, 1, 4, -2;
  B << 1, -1, 1, -1;
  std::pair<Eigen::MatrixXd, Eigen::MatrixXd> sys_discrete;
  sys_discrete = control::continuous_to_discrete(A, B, 0.005);
  Eigen::MatrixXd Ad_expected(2, 2);
  Ad_expected << 1.005062521587947, 0.004987645496221,
                 0.019950581984884, 0.990099585099284;
  Eigen::MatrixXd Bd_expected(2, 2);
  Bd_expected << 0.005025083542084, -0.005025083542084,
                 0.005025083542084, -0.005025083542084;

  EXPECT_TRUE(sys_discrete.first.isApprox(Ad_expected));
  EXPECT_TRUE(sys_discrete.second.isApprox(Bd_expected));
}

