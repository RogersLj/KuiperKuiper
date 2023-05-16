#ifndef KUIPER_INFER_INCLUDE_DATA_LOAD_DATA_HPP
#define KUIPER_INFER_INCLUDE_DATA_LOAD_DATA_HPP
#include <armadillo>
#include <string>

namespace kuiper_infer {

class CSVDataLoader {
public:

// 从csv文件里读取','隔开的数据并存入fmat
    static arma::fmat LoadData(const std::string &file_path, char split_char = ',');


private:
// 文件里fmat的大小
    static std::pair<size_t, size_t> GetMatrixSize(std::ifstream &file, char split_char);

};

}

#endif //KUIPER_INFER_INCLUDE_DATA_LOAD_DATA_HPP