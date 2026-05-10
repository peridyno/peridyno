#include <iostream>
#include <string>
#include <vector>
#include <cctype>
#include <algorithm>
#include <sstream>

namespace DescriptionHelper 
{
    bool iequals(const std::string& a, const std::string& b) {
        return std::equal(a.begin(), a.end(), b.begin(), b.end(),
            [](char c1, char c2) { return std::tolower(c1) == std::tolower(c2); });
    }

    std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\n\r\f\v");
        if (start == std::string::npos) return "";
        size_t end = s.find_last_not_of(" \t\n\rf\v");
        return s.substr(start, end - start + 1);
    }

    std::string collapseSpaces(const std::string& s) {
        std::string result;
        bool inSpace = false;
        for (char ch : s) {
            if (std::isspace(ch)) {
                if (!inSpace && !result.empty()) {
                    result += ' ';
                    inSpace = true;
                }
            }
            else {
                result += ch;
                inSpace = false;
            }
        }
        return trim(result);
    }

    bool parseQtStyleDescriptionRobust(const std::string& input,
        std::string& cleanDescription,
        bool& IsVLayout,
        bool& onlyDetail) {
        IsVLayout = true;
        onlyDetail = false;
        std::string working = input;
        bool foundAny = false;

        auto findQtStylePos = [](const std::string& str, size_t start = 0) -> size_t {
            const std::string marker = "qtstyle";
            size_t pos = start;
            while (pos < str.length()) {
                size_t found = str.find_first_of("Qq", pos);
                if (found == std::string::npos) break;
                if (str.length() - found < marker.length()) break;
                if (iequals(str.substr(found, marker.length()), marker)) {
                    size_t after = found + marker.length();
                    while (after < str.length() && std::isspace(str[after])) ++after;
                    if (after < str.length() && str[after] == '(') {
                        return found;
                    }
                    else {
                        pos = found + 1;
                        continue;
                    }
                }
                pos = found + 1;
            }
            return std::string::npos;
        };

        while (true) {
            size_t startPos = findQtStylePos(working);
            if (startPos == std::string::npos) break;

            size_t nameEnd = startPos + std::string("qtstyle").length();
            size_t parenPos = nameEnd;
            while (parenPos < working.length() && std::isspace(working[parenPos])) ++parenPos;
            if (parenPos >= working.length() || working[parenPos] != '(') break;
            size_t openParen = parenPos;

            int depth = 1;
            size_t closeParen = openParen + 1;
            while (closeParen < working.length() && depth > 0) {
                if (working[closeParen] == '(') depth++;
                else if (working[closeParen] == ')') depth--;
                if (depth == 0) break;
                ++closeParen;
            }
            if (depth != 0) break;

            std::string params = working.substr(openParen + 1, closeParen - openParen - 1);
            std::vector<std::string> tokens;
            std::stringstream ss(params);
            std::string token;
            while (std::getline(ss, token, ',')) {
                std::string trimmed = trim(token);
                if (!trimmed.empty()) tokens.push_back(trimmed);
            }

            for (const auto& t : tokens) {
                if (iequals(t, "VLayout")) {
                    IsVLayout = true;
                }
                else if (iequals(t, "HLayout")) {
                    IsVLayout = false;
                }
                else if (iequals(t, "OnlyDetail")) {
                    onlyDetail = true;
                }
            }

            working.erase(startPos, closeParen - startPos + 1);
            foundAny = true;
        }

        cleanDescription = collapseSpaces(working);
        return foundAny;
    }

}
