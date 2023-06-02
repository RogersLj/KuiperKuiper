#include "data/tensor.hpp"
#include "data/tensor_util.hpp"
#include <gtest/gtest.h>
#include <armadillo>
#include <glog/logging.h>


TEST(test_tensor, tensor_init1) {
  using namespace kuiper_infer;
  Tensor<float> f1(3, 224, 224);
  ASSERT_EQ(f1.channels(), 3);
  ASSERT_EQ(f1.rows(), 224);
  ASSERT_EQ(f1.cols(), 224);
  ASSERT_EQ(f1.size(), 224 * 224 * 3);
}

TEST(test_tensor, tensor_init2) {
  using namespace kuiper_infer;
  Tensor<float> f1(std::vector<uint32_t>{3, 224, 224});
  ASSERT_EQ(f1.channels(), 3);
  ASSERT_EQ(f1.rows(), 224);
  ASSERT_EQ(f1.cols(), 224);
  ASSERT_EQ(f1.size(), 224 * 224 * 3);
}

TEST(test_tensor, copy_construct1) {
  using namespace kuiper_infer;
  Tensor<float> f1(3, 224, 224);
  f1.Rand();
  Tensor<float> f2(f1);
  ASSERT_EQ(f2.channels(), 3);
  ASSERT_EQ(f2.rows(), 224);
  ASSERT_EQ(f2.cols(), 224);

  ASSERT_TRUE(arma::approx_equal(f2.data(), f1.data(), "absdiff", 1e-4));
}

TEST(test_tensor, copy_construct2) {
  using namespace kuiper_infer;
  Tensor<float> f1(3, 2, 1);
  Tensor<float> f2(3, 224, 224);
  f2.Rand();
  f1 = f2;
  ASSERT_EQ(f1.channels(), 3);
  ASSERT_EQ(f1.rows(), 224);
  ASSERT_EQ(f1.cols(), 224);

  ASSERT_TRUE(arma::approx_equal(f2.data(), f1.data(), "absdiff", 1e-4));
}

TEST(test_tensor, copy_construct3) {
  using namespace kuiper_infer;
  Tensor<float> f1(3, 2, 1);
  Tensor<float> f2(std::vector<uint32_t>{3, 224, 224});
  f2.Rand();
  f1 = f2;
  ASSERT_EQ(f1.channels(), 3);
  ASSERT_EQ(f1.rows(), 224);
  ASSERT_EQ(f1.cols(), 224);

  ASSERT_TRUE(arma::approx_equal(f2.data(), f1.data(), "absdiff", 1e-4));
}

TEST(test_tensor, move_construct1) {
  using namespace kuiper_infer;
  Tensor<float> f1(3, 2, 1);
  Tensor<float> f2(3, 224, 224);
  f1 = std::move(f2);
  ASSERT_EQ(f1.channels(), 3);
  ASSERT_EQ(f1.rows(), 224);
  ASSERT_EQ(f1.cols(), 224);

  ASSERT_EQ(f2.data().memptr(), nullptr);
}

TEST(test_tensor, move_construct2) {
  using namespace kuiper_infer;
  Tensor<float> f2(3, 224, 224);
  Tensor<float> f1(std::move(f2));
  ASSERT_EQ(f1.channels(), 3);
  ASSERT_EQ(f1.rows(), 224);
  ASSERT_EQ(f1.cols(), 224);

  ASSERT_EQ(f2.data().memptr(), nullptr);
}

TEST(test_tensor, set_data) {
  using namespace kuiper_infer;
  Tensor<float> f2(3, 224, 224);
  arma::fcube cube1(224, 224, 3);
  cube1.randn();
  f2.set_data(cube1);

  ASSERT_TRUE(arma::approx_equal(f2.data(), cube1, "absdiff", 1e-4));
}

TEST(test_tensor, data) {
  using namespace kuiper_infer;
  Tensor<float> f2(3, 224, 224);
  f2.Fill(1.f);
  arma::fcube cube1(224, 224, 3);
  cube1.fill(1.);
  f2.set_data(cube1);

  ASSERT_TRUE(arma::approx_equal(f2.data(), cube1, "absdiff", 1e-4));
}

TEST(test_tensor, empty) {
  using namespace kuiper_infer;
  Tensor<float> f2;
  ASSERT_EQ(f2.empty(), true);

  Tensor<float> f3(3, 3, 3);
  ASSERT_EQ(f3.empty(), false);
}

TEST(test_tensor, transform1) {
  using namespace kuiper_infer;

  Tensor<float> f3(3, 3, 3);
  ASSERT_EQ(f3.empty(), false);
  f3.Transform([](const float& value) { return 1.f; });
  for (int i = 0; i < f3.size(); ++i) {
    ASSERT_EQ(f3.index(i), 1.f);
  }
}

TEST(test_tensor, transform2) {
  using namespace kuiper_infer;

  Tensor<float> f3(3, 3, 3);
  ASSERT_EQ(f3.empty(), false);
  f3.Fill(1.f);
  f3.Transform([](const float& value) { return value * 2.f; });
  for (int i = 0; i < f3.size(); ++i) {
    ASSERT_EQ(f3.index(i), 2.f);
  }
}

TEST(test_tensor, raw_ptr) {
  using namespace kuiper_infer;

  Tensor<float> f3(3, 3, 3);
  ASSERT_EQ(f3.raw_ptr(), f3.data().mem);
}

TEST(test_tensor, index1) {
  using namespace kuiper_infer;
  Tensor<float> f3(3, 3, 3);
  ASSERT_EQ(f3.empty(), false);
  std::vector<float> values;
  for (int i = 0; i < 27; ++i) {
    values.push_back(1);
  }
  f3.Fill(values);
  for (int i = 0; i < 27; ++i) {
    ASSERT_EQ(f3.index(i), 1);
  }
}

TEST(test_tensor, index2) {
  using namespace kuiper_infer;
  Tensor<float> f3(3, 3, 3);
  f3.index(3) = 4;
  ASSERT_EQ(f3.index(3), 4);
}

TEST(test_tensor, flatten1) {
  using namespace kuiper_infer;

  Tensor<float> f3(3, 3, 3);
  std::vector<float> values;
  for (int i = 0; i < 27; ++i) {
    values.push_back(float(i));
  }
  f3.Fill(values);
  f3.Flatten(false);
  ASSERT_EQ(f3.channels(), 1);
  ASSERT_EQ(f3.rows(), 1);
  ASSERT_EQ(f3.cols(), 27);
  ASSERT_EQ(f3.index(0), 0);
  ASSERT_EQ(f3.index(1), 3);
  ASSERT_EQ(f3.index(2), 6);

  ASSERT_EQ(f3.index(3), 1);
  ASSERT_EQ(f3.index(4), 4);
  ASSERT_EQ(f3.index(5), 7);

  ASSERT_EQ(f3.index(6), 2);
  ASSERT_EQ(f3.index(7), 5);
  ASSERT_EQ(f3.index(8), 8);
}

TEST(test_tensor, flatten2) {
  using namespace kuiper_infer;

  Tensor<float> f3(3, 3, 3);
  std::vector<float> values;
  for (int i = 0; i < 27; ++i) {
    values.push_back(float(i));
  }
  f3.Fill(values);
  f3.Flatten(true);
  for (int i = 0; i < 27; ++i) {
    ASSERT_EQ(f3.index(i), i);
  }
}

TEST(test_tensor, fill1) {
  using namespace kuiper_infer;

  Tensor<float> f3(3, 3, 3);
  std::vector<float> values;
  for (int i = 0; i < 27; ++i) {
    values.push_back(float(i));
  }
  f3.Fill(values);
  int index = 0;
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < f3.rows(); ++i) {
      for (int j = 0; j < f3.cols(); ++j) {
        ASSERT_EQ(f3.at(c, i, j), index);
        index += 1;
      }
    }
  }
}

TEST(test_tensor, fill2_colmajor1) {
  using namespace kuiper_infer;

  Tensor<float> f3(3, 3, 3);
  std::vector<float> values;
  for (int i = 0; i < 27; ++i) {
    values.push_back(float(i));
  }
  f3.Fill(values, false);
  for (int i = 0; i < 27; ++i) {
    ASSERT_EQ(values.at(i), f3.index(i));
  }
}

TEST(test_tensor, fill2_colmajor2) {
  using namespace kuiper_infer;

  Tensor<float> f3(1, 27, 1);
  std::vector<float> values;
  for (int i = 0; i < 27; ++i) {
    values.push_back(float(i));
  }
  f3.Fill(values, false);
  for (int i = 0; i < 27; ++i) {
    ASSERT_EQ(values.at(i), f3.index(i));
  }
}

TEST(test_tensor, add2) {
  using namespace kuiper_infer;
  const auto& f1 = std::make_shared<Tensor<float>>(3, 224, 224);
  f1->Fill(1.f);
  const auto& f2 = std::make_shared<Tensor<float>>(3, 1, 1);
  f2->Fill(2.f);
  const auto& f3 = TensorElementAdd(f2, f1);
  for (int i = 0; i < f3->size(); ++i) {
    ASSERT_EQ(f3->index(i), 3.f);
  }
}