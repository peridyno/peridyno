#include "GlslShaderParser.h"

using namespace glslang;

void GlslShaderParser::ParseShader(void *assetManager, const std::string &fileName,
                                   std::string &shaderCode,
                                   const std::map<std::string, std::string> &macros,
                                   bool writeSpv) {
    EShLanguage stage = FindStage(fileName);
    std::string fileText = ReadFile(assetManager, fileName);
    WriteMacro(fileText, macros);
    const char *fileNames[1] = {fileName.c_str()};
    const char *fileTexts[1] = {fileText.c_str()};

    TShader shader(stage);

    EShTargetClientVersion clientVersion = EShTargetVulkan_1_1;
    EShTargetLanguageVersion languageVersion = EShTargetSpv_1_3;
    if (stage == EShLangRayGen || stage == EShLangClosestHit || stage == EShLangMiss) {
        clientVersion = EShTargetVulkan_1_2;
        languageVersion = EShTargetSpv_1_5;
    }

    shader.setEnvInput(EShSourceGlsl, stage, EShClientVulkan, defaultVersion);
    shader.setEnvClient(EShClientVulkan, clientVersion);
    shader.setEnvTarget(EShTargetSpv, languageVersion);
    shader.setStringsWithLengthsAndNames(fileTexts, nullptr, fileNames, 1);
    shader.parse(&resource, defaultVersion, false, messages, includer);

    TProgram program;
    program.addShader(&shader);
    program.link(messages);

    std::vector<unsigned int> spirv;
    GlslangToSpv(*program.getIntermediate(stage), spirv);
    shaderCode.resize(spirv.size() * sizeof(unsigned int));
    for (size_t i = 0; i < spirv.size(); ++i) {
        memcpy(&shaderCode[0] + i * sizeof(unsigned int), (const char *) &spirv[i],
               sizeof(unsigned int));
    }

    if (writeSpv) {
        OutputSpvBin(spirv, (fileName + ".spv").c_str());
    }
}

void GlslShaderParser::ParseShader(void *assetManager, const std::string &fileName,
	std::string &shaderCode,
	const std::map<std::string, std::string> &macros, const std::string &MD5EnCode,
	bool writeSpv) {
	EShLanguage stage = FindStage(fileName);
	std::string fileText = ReadFile(assetManager, fileName);
	WriteMacro(fileText, macros);
	const char *fileNames[1] = { fileName.c_str() };
	const char *fileTexts[1] = { fileText.c_str() };

	TShader shader(stage);

	EShTargetClientVersion clientVersion = EShTargetVulkan_1_1;
	EShTargetLanguageVersion languageVersion = EShTargetSpv_1_3;
	if (stage == EShLangRayGen || stage == EShLangClosestHit || stage == EShLangMiss) {
		clientVersion = EShTargetVulkan_1_2;
		languageVersion = EShTargetSpv_1_5;
	}

	shader.setEnvInput(EShSourceGlsl, stage, EShClientVulkan, defaultVersion);
	shader.setEnvClient(EShClientVulkan, clientVersion);
	shader.setEnvTarget(EShTargetSpv, languageVersion);
	shader.setStringsWithLengthsAndNames(fileTexts, nullptr, fileNames, 1);
	shader.parse(&resource, defaultVersion, false, messages, includer);

	TProgram program;
	program.addShader(&shader);
	program.link(messages);

	std::vector<unsigned int> spirv;
	GlslangToSpv(*program.getIntermediate(stage), spirv);
	shaderCode.resize(spirv.size() * sizeof(unsigned int));
	for (size_t i = 0; i < spirv.size(); ++i) {
		memcpy(&shaderCode[0] + i * sizeof(unsigned int), (const char *)&spirv[i],
			sizeof(unsigned int));
	}

	if (writeSpv) {
		OutputSpvBin(spirv, (MD5EnCode).c_str());
	}
}

GlslShaderParser::GlslShaderParser() { InitializeProcess(); }

GlslShaderParser::~GlslShaderParser() { FinalizeProcess(); }

EShLanguage GlslShaderParser::FindStage(const std::string &fileName) {
    size_t len = fileName.rfind('.');
    std::string ext = fileName.substr(len + 1);
    if (ext == "glsl" || ext == "hlsl") {
        size_t trueLen = fileName.rfind('.', len - 1);
        ext = fileName.substr(trueLen + 1, len - trueLen - 1);
    }

    if (stages.find(ext) == stages.end()) {
        return LAST_ELEMENT_MARKER(EShLangCount);
    }
    return stages[ext];
}

std::string GlslShaderParser::ReadFile(void *assetManager, const std::string &fileName) {
#if defined(__ANDROID__)
    AAsset *asset = AAssetManager_open(reinterpret_cast<AAssetManager*>(assetManager), fileName.c_str(), AASSET_MODE_STREAMING);
    assert(asset);
    size_t size = AAsset_getLength(asset);
    assert(size > 0);
    std::string fileText(size, '\0');
    AAsset_read(asset, &fileText[0], size);
    AAsset_close(asset);
#else
    std::ifstream is(fileName, std::ios::in | std::ios::ate);
    if (!is.is_open()) {
        std::cerr << fileName << " does not open" << std::endl;
        return "";
    }
    size_t size = is.tellg();
    is.seekg(0, std::ios::beg);
    std::string fileText(size, '\0');
    is.read(&fileText[0], size);
    is.close();
#endif
    return fileText;
}

void GlslShaderParser::WriteMacro(std::string &fileText,
                                  const std::map<std::string, std::string> &macros) {
    for (const auto &macro : macros) {
        std::string keyword = prefix + macro.first;
        if (fileText.find(keyword) != -1) {
            auto pos = fileText.find(keyword);
            auto newlinePos = std::min(fileText.find('\n', pos), (fileText.find('\r', pos)));
            auto len = newlinePos - pos - keyword.length() - 1; // subtract the length of space (=1)
            fileText.replace(pos + keyword.length() + 1, len, macro.second);
        }
    }
}
