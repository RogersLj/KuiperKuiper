#include <glog/logging.h>
#include <gtest/gtest.h>
#include "data/load_data.hpp"

TEST(test_load_data, load_csv1) {
    using namespace kuiper_infer;

    const std::string& file_path = "../tmp/data1.csv";
    const arma::fmat &data = CSVDataLoader::LoadData(file_path, ',');

    uint32_t index = 1;
    uint32_t rows = data.n_rows;
    uint32_t cols = data.n_cols;
    ASSERT_EQ(rows, 3); 
    ASSERT_EQ(cols, 5); 
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
        ASSERT_EQ(data.at(r, c), index);
        index += 1;
        }   
    }
}