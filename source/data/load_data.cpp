#include "data/load_data.hpp"
#include <string>
#include <fstream>
#include <armadillo>
#include <utility>
#include <glog/logging.h>

namespace kuiper_infer {

arma::fmat CSVDataLoader::LoadData(const std::string &file_path, const char split_char) {
    arma::fmat data;
    if (file_path.empty()) { // 检查文件路径是否为空 
        LOG(ERROR) << "CSV file path is empty: " << file_path;
        return data;
    }

    std::ifstream infile(file_path);

    if (!infile.good()) {
        LOG(ERROR) << "File open failed: " << file_path;
        return data;
    }

    size_t rows = 0, cols = 0;
    std::tie(rows, cols) = CSVDataLoader::GetMatrixSize(infile, split_char);
    data.zeros(rows, cols);

    for (size_t row = 0; infile.good() && row < rows; ++row) {
        std::string line_Str;
        std::getline(infile, line_Str);

        if (line_Str.empty()) {
            break;
        }

        std::stringstream line_stream(line_Str);

        size_t col = 0;

        while (line_stream.good() && col < cols) {
            std::string cur_token;
            std::getline(line_stream, cur_token, split_char);

            try {
                data(row, col++) = std::stof(cur_token);
            } catch (std::exception& e) {
                DLOG(ERROR) << "Parse CSV File meet error: " << e.what() << " row:" << row << " col:" << col;
                }
        }

        // 检查当前行是否有超出指定列数的元素
        CHECK(col <= cols) << "There are excessive elements on the column";
    }
    // 检查当前列是否有超出指定行数的元素
    CHECK(data.n_rows == rows) << "There are excessive elements on the row";

    return data;
}

std::pair<size_t, size_t> CSVDataLoader::GetMatrixSize(std::ifstream &file, const char split_char) {
    file.clear();

    size_t rows = 0;
    size_t cols = 0;

    const std::ifstream::pos_type start_pos = file.tellg();

    std::string cur_token; // 每个需要存储的数值
    std::string line_str; // 每次读入一行
    std::stringstream line_stream; // 从中解析数值

    while (std::getline(file, line_str)) {
        if (line_str.empty()) {
            break; // 读到文件结束
        }

        line_stream.clear();
        line_stream.str(line_str);
        size_t line_cols = 0; // 当前行的列数
        while (std::getline(line_stream, cur_token, split_char)) {
            ++line_cols;
        }

        if (line_cols > cols) {
            cols = line_cols;
        }

        ++rows;
    }
    
    file.clear(); // 清空文件流状态
    file.seekg(start_pos); // 将文件指针移回起始位置
    return {rows, cols};
}

}
