#ifndef KUIPER_INFER_PARSER_PARSE_EXPRESSION_HPP
#define KUIPER_INFER_PARSER_PARSE_EXPRESSION_HPP

#include <string>
#include <utility>
#include <vector>
#include <memory>

namespace kuiper_infer {
    
enum class TokenType {
    TokenUnknown = -1,
    TokenNumber = 0,
    TokenComma = 1,
    TokenAdd = 2,
    TokenMul = 3,
    TokenLeftBracket = 4,
    TokenRightBracket = 5,
};

struct Token {
    TokenType token_type_ = TokenType::TokenUnknown;
    int32_t start_pos = 0;
    int32_t end_pos = 0;
    Token(TokenType token_type, int32_t start_pos, int32_t end_pos) : token_type_(token_type), start_pos(start_pos), end_pos(end_pos) {} 
};

struct TokenNode {

    int32_t num_index = -1; // 为正时是对应的数字，为负的时候是运算符
    std::shared_ptr<TokenNode> left = nullptr;
    std::shared_ptr<TokenNode> right = nullptr;
    TokenNode(int32_t num_index, std::shared_ptr<TokenNode> left, std::shared_ptr<TokenNode> right) : num_index(num_index), left(std::move(left)), right(std::move(right)) {}
    TokenNode() = default;
};

class ExpressionParser {
    
public:
    explicit ExpressionParser(std::string expression) : expression_(expression) {};

    void Tokenizer(bool need_tokenizer = false);

    std::vector<std::shared_ptr<TokenNode>> Generate(); // 构建计算图返回所有token节点的左右根存储

    const std::vector<Token>& tokens() const; // 返回所有token
    const std::vector<std::string>& token_strs() const; // 返回所有token的字符串

    static void PrintNodes(const std::shared_ptr<TokenNode> &node);


private:
    std::shared_ptr<TokenNode> Generate_(int32_t &index); // 构建计算图并返回index对应的token节点
    std::vector<Token> tokens_;
    std::vector<std::string> token_strs_;
    std::string expression_;

};

}



#endif