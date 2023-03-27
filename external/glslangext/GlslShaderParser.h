//
// a simple glsl shader parser wrapping around glslang
//
#pragma once

#include "glslang/glslang/Public/ShaderLang.h"
#include "glslang/SPIRV/GlslangToSpv.h"
#include "glslang/StandAlone/DirStackFileIncluder.h"
#include "glslang/StandAlone/ResourceLimits.h"

#if defined(__ANDROID__)
#include <android/asset_manager.h>
#else

#include <iostream>
#include <fstream>

#endif

class GlslShaderParser {
public:
    void ParseShader(void *assetManager, const std::string &fileName,
                     std::string &shaderCode, const std::map<std::string, std::string> &macros,
                     bool writeSpv = false);

	void ParseShader(void *assetManager, const std::string &fileName,
		std::string &shaderCode, const std::map<std::string, std::string> &macros, const std::string &MD5EnCode,
		bool writeSpv = true);

    GlslShaderParser();

    ~GlslShaderParser();

private:
    int defaultVersion = 110; // use 100 for ES environment, 110 for desktop
    DirStackFileIncluder includer;
    TBuiltInResource resource = glslang::DefaultTBuiltInResource;
    EShMessages messages = EShMsgDefault;
    const std::string prefix = "#define ";

    std::unordered_map<std::string, EShLanguage> stages = {
            {"vert",  EShLangVertex},
            {"tesc",  EShLangTessControl},
            {"tese",  EShLangTessEvaluation},
            {"geom",  EShLangGeometry},
            {"frag",  EShLangFragment},
            {"comp",  EShLangCompute},
            {"rgen",  EShLangRayGen},
            {"rint",  EShLangIntersect},
            {"rahit", EShLangAnyHit},
            {"rchit", EShLangClosestHit},
            {"rmiss", EShLangMiss},
            {"rcall", EShLangCallable},
            {"mesh",  EShLangMeshNV},
            {"task",  EShLangTaskNV}
    };

    EShLanguage FindStage(const std::string &fileName);

    static std::string ReadFile(void* assetManager, const std::string &fileName);

    void WriteMacro(std::string &fileText, const std::map<std::string, std::string> &macros);
};