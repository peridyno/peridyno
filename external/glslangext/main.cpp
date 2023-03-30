#include <iostream>
#include <map>
#include <string>
#include "GlslShaderParser.h"

int main() {

    std::string shaderCode;
    GlslShaderParser parser;
    std::map<std::string, std::string> macros = {};
    bool writeSpv = false;
    std::string fileName = "shaders/glsl/core/Sort.comp";

#if defined(__ANDROID__)
    // AAssetManager* assetManager = androidApp->activity->assetManager; // app use
    AAssetManager* assetManager = nullptr;
    parser.ParseShader(assetManager, fileName, shaderCode, macros, writeSpv);
#else
    parser.ParseShader(nullptr, fileName, shaderCode, macros, writeSpv);
#endif

    return 0;
}
