#include "parser/parse_expression.hpp"
#include <algorithm>
#include <cctype>
#include <stack>
#include <utility>
#include <glog/logging.h>

namespace kuiper_infer {


// 遍历整个ast，以左右根的顺序保存所有TokenNode
void ReversePolish(const std::shared_ptr<TokenNode>& root, std::vector<std::shared_ptr<TokenNode>>& reverse_polish) {
    
    if (!root) {
        return;
    }

    ReversePolish(root->left, reverse_polish);
    ReversePolish(root->right, reverse_polish);

    reverse_polish.push_back(root);
}

std::string RemovingSpaces(const std::string& expression) {
    std::string result = expression;

    size_t index = 0;

    while (index < expression.size() && std::isspace(expression.at(index))) {
        ++index;
    }

    result.erase(0, index);
    return result;
} 

void ExpressionParser::Tokenizer(bool need_tokenizer) {
    if (!need_tokenizer && !this->tokens_.empty()) {
        return; // 不需要tokenize了
    }

    CHECK(!expression_.empty()) << "The input expression is empty.";

    std::string expression = RemovingSpaces(expression_);
    CHECK(!expression.empty()) << "The input expression is empty.";

    for (int32_t i = 0; i < expression.size();) {
        auto c = expression.at(i);

        if (c == 'a') {
            // 默认一定是add
            CHECK(i + 1 < expression.size() && expression.at(i + 1) == 'd') << "The input expression is invalid.";
            CHECK(i + 2 < expression.size() && expression.at(i + 2) == 'd') << "The input expression is invalid.";

            Token token(TokenType::TokenAdd, i, i + 3);
            tokens_.push_back(token);
            token_strs_.push_back("add");
            i += 3;
        } else if (c == 'm') {
            // 默认一定是mul
            CHECK(i + 1 < expression.size() && expression.at(i + 1) == 'u') << "The input expression is invalid.";
            CHECK(i + 2 < expression.size() && expression.at(i + 2) == 'l') << "The input expression is invalid.";
            Token token(TokenType::TokenMul, i, i + 3);
            tokens_.push_back(token);
            token_strs_.push_back("mul");
            i += 3;
        } else if (c == '(') {
            Token token = Token(TokenType::TokenLeftBracket, i, i + 1);
            tokens_.push_back(token);
            token_strs_.push_back("(");
            i += 1;
        } else if (c == ')') {
            Token token = Token(TokenType::TokenRightBracket, i, i + 1);
            tokens_.push_back(token);
            token_strs_.push_back(")");
            i += 1;
        } else if (c == ',') {
            Token token = Token(TokenType::TokenComma, i, i + 1);
            tokens_.push_back(token);
            token_strs_.push_back(",");
            i += 1;
        } else if (c == '@') {
            CHECK(i + 1 < expression.size() && isdigit(expression.at(i + 1))) << "The input expression is invalid.";
            int32_t j = i + 1;
            while (j < expression.size() && isdigit(expression.at(j))) {
                j ++;
                if (!isdigit(expression.at(j))) {
                    break;
                }
            }

            Token token(TokenType::TokenNumber, i + 1, j);
            tokens_.push_back(token);
            token_strs_.push_back(expression.substr(i + 1, j));

            i = j;
        } else {
            LOG(FATAL) << "The input expression is invalid.";
        }
    }
}


std::shared_ptr<TokenNode> ExpressionParser::Generate_(int32_t &index) {
    CHECK(index < this->tokens_.size());
    auto current_token = this->tokens_.at(index);

    // 第一次肯定是add or mul 
    CHECK(current_token.token_type_ == TokenType::TokenNumber || current_token.token_type_ == TokenType::TokenAdd || current_token.token_type_ == TokenType::TokenMul);

    if (current_token.token_type_ == TokenType::TokenNumber) {
        // 取出数字，左右子节点为空
        const std::string &str_num = this->expression_.substr(current_token.start_pos, current_token.end_pos - current_token.start_pos);

        // LOG(INFO) << "number is -------- " << str_num;

        return std::make_shared<TokenNode>(std::stoi(str_num), nullptr, nullptr);
    } else if (current_token.token_type_ == TokenType::TokenAdd || current_token.token_type_ == TokenType::TokenMul) {

        // LOG(INFO) << "op is --------" << token_strs_.at(index);

        std::shared_ptr<TokenNode> current_node = std::make_shared<TokenNode>();

        current_node->num_index = -int(current_token.token_type_);
        // 为正时是对应的数字，为负的时候是运算符 type

        index += 1;
        // 递归生成左右根
        CHECK(index < this->tokens_.size());
        CHECK(this->tokens_.at(index).token_type_ == TokenType::TokenLeftBracket);

        index += 1; // 跳过左括号
        CHECK(index < this->tokens_.size());
        const auto left_token = this->tokens_.at(index);
        // 存入的时候就只存的数字没有存@
        // 可能会有嵌套
        if (left_token.token_type_ == TokenType::TokenNumber || left_token.token_type_ == TokenType::TokenAdd || left_token.token_type_ == TokenType::TokenMul) {
            current_node->left = Generate_(index);
        } else {
            LOG(FATAL) << "The input expression is invalid.";
        }

        // 遇到逗号
        index += 1;
        CHECK(index < this->tokens_.size());
        CHECK(this->tokens_.at(index).token_type_ == TokenType::TokenComma);

        // 跳过逗号
        index += 1;
        CHECK(index < this->tokens_.size());

        const auto right_token = this->tokens_.at(index);
        if (right_token.token_type_ == TokenType::TokenNumber || right_token.token_type_ == TokenType::TokenAdd || right_token.token_type_ == TokenType::TokenMul) {
            current_node->right = Generate_(index);
        } else {
            LOG(FATAL) << "The input expression is invalid.";
        }

        index += 1;
        CHECK(index < this->tokens_.size());
        // 跳过右括号
        CHECK(this->tokens_.at(index).token_type_ == TokenType::TokenRightBracket);

        return current_node;
    } else {
        LOG(FATAL) << "The input expression is invalid.";
    }
}
    

std::vector<std::shared_ptr<TokenNode>> ExpressionParser::Generate() {
    if (this->tokens_.empty()) {
        this->Tokenizer(true);
    } // 需要进行分词

    int index = 0; // 首先创建根节点
    std::shared_ptr<TokenNode> root = Generate_(index);
    CHECK(root != nullptr);
    CHECK(index == tokens_.size() - 1); // 是否创建完所有token

    std::vector<std::shared_ptr<TokenNode>> reverse_polish;
    // 左右根保存
    ReversePolish(root, reverse_polish);

    return reverse_polish;
}


const std::vector<std::string>& ExpressionParser::token_strs() const {
    return this->token_strs_;
}


const std::vector<Token>& ExpressionParser::tokens() const {
    return this->tokens_;
}

void ExpressionParser::PrintNodes(const std::shared_ptr<TokenNode> &node) {
    if (!node) return;

    PrintNodes(node->left);

    PrintNodes(node->right);

    if (node->num_index >= 0) {
        LOG(INFO) << "num index: " << node->num_index;
    } else if (node->num_index == -int(kuiper_infer::TokenType::TokenAdd)) {
        LOG(INFO) << "add";
    } else if (node->num_index == -int(kuiper_infer::TokenType::TokenMul)) {
        LOG(INFO) << "mul";
    }
}

}