#include "GLDigitNumVisualModule.h"
#include "GLRenderEngine.h"
#include <Utility.h>
#include <glad/glad.h>
#include <iostream>

#include "digit.vert.h"
#include "digit.frag.h"
#include "digit.geom.h"
namespace dyno
{
    IMPLEMENT_CLASS(GLDigitNumVisualModule)

        GLDigitNumVisualModule::GLDigitNumVisualModule()
        : mNumPoints(0)
    {
        this->setName("digit_renderer");
        this->varDigitScale()->setRange(0.0001f, 0.1f);
    }

    GLDigitNumVisualModule::~GLDigitNumVisualModule()
    {
        this->releaseGL();
    }

    bool GLDigitNumVisualModule::initializeGL()
    {
        mPosition.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

        mVertexArray.create();
        mVertexArray.bindVertexBuffer(&mPosition, 0, 3, GL_FLOAT, 0, 0, 0);

        //mShaderProgram = Program::createProgram("./Shader Sources/digit.vert", "./Shader Sources/digit.frag", "./Shader Sources/digit.geom");
        // create shader program
        mShaderProgram = Program::createProgramSPIRV(
            DIGIT_VERT, sizeof(DIGIT_VERT),
            DIGIT_FRAG, sizeof(DIGIT_FRAG),
            DIGIT_GEOM, sizeof(DIGIT_GEOM));

        if (!loadDigitTextures()) {
            std::cerr << "Failed to load digit textures!" << std::endl;

            return false;
        }
        else
        {
            std::cerr << "Successfully loaded texture!" << std::endl;
        }

        mUniformBlock.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


        glCheckError();
        return true;
    }

    bool GLDigitNumVisualModule::loadDigitTextures()
    {
        glGenTextures(1, &mDigitTexture);
        glBindTexture(GL_TEXTURE_2D, mDigitTexture);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_set_flip_vertically_on_load(true);
        std::string path = getAssetPath() + "digitNums/textures/v1_0123456789.png";
        int width, height, nrChannels;
        unsigned char* data = stbi_load(path.c_str(), &width, &height, &nrChannels, 0);
        if (data) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);
        }
        else {
            std::cout << "Trying to load texture from: " << path << std::endl;
        }
        stbi_image_free(data);

        mShaderProgram->use();
        mShaderProgram->setFloat("uDigitScale", this->varDigitScale()->getValue());
        mShaderProgram->setVec2("uDigitOffset", this->varDigitOffset()->getValue());
        mShaderProgram->setFloat("uDigitWidth", 0.1f);

        glCheckError();
        return true;
    }

    void GLDigitNumVisualModule::updateImpl()
    {
        auto pPointSet = this->inPointSet()->getDataPtr();
        auto points = pPointSet->getPoints();
        mPosition.load(points);


    }

    void GLDigitNumVisualModule::updateGL()
    {
        mNumPoints = mPosition.count();
        if (mNumPoints == 0) return;

        mPosition.updateGL();
        mVertexArray.bindVertexBuffer(&mPosition, 0, 3, GL_FLOAT, 0, 0, 0);


    }

    void GLDigitNumVisualModule::paintGL(const RenderParams& rparams)
    {
        if (mNumPoints == 0) return;

        mUniformBlock.load((void*)&rparams, sizeof(RenderParams));
        mUniformBlock.bindBufferBase(0);

        mShaderProgram->use();
        mShaderProgram->setFloat("uDigitScale", this->varDigitScale()->getValue());
        mShaderProgram->setInt("uDigitSampler", 0);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, mDigitTexture);
        mVertexArray.bind();
        glDrawArrays(GL_POINTS, 0, mNumPoints);

        glCheckError();
    }

    void GLDigitNumVisualModule::releaseGL()
    {
        if (mShaderProgram) {
            mShaderProgram->release();
            delete mShaderProgram;
            mShaderProgram = nullptr;
        }

        if (mDigitTexture) {
            glDeleteTextures(1, &mDigitTexture);
            mDigitTexture = 0;
        }

        mPosition.release();
        mVertexArray.release();
        mUniformBlock.release();
    }
}