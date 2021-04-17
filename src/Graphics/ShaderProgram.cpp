/*
* @file shader_program.cpp
* @Brief Class ShaderProgram used to generate openGL program from shaders
* @author Wei Chen
*
* This file is part of PhysIKA, a versatile physics simulation library.
* Copyright (C) 2013- PhysIKA Group.
*
* This Source Code Form is subject to the terms of the GNU General Public License v2.0.
* If a copy of the GPL was not distributed with this file, you can obtain one at:
* http://www.gnu.org/licenses/gpl-2.0.html
*
*/

#include "GlewHelper.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>


#include "ShaderProgram.h"

namespace dyno {
/*
 * glslPrintShaderLog, output error message if fail to compile shader sources
 */

void glslPrintShaderLog(GLuint obj)
{
    int infologLength = 0;
    int charsWritten = 0;
    char *infoLog;

    GLint result;
    glGetShaderiv(obj, GL_COMPILE_STATUS, &result);

    // only print log if compile fails
    if (result == GL_FALSE)
    {
        glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &infologLength);

        if (infologLength > 1)
        {
            infoLog = (char *)malloc(infologLength);
            glGetShaderInfoLog(obj, infologLength, &charsWritten, infoLog);
            printf("%s\n", infoLog);
            free(infoLog);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
ShaderProgram::ShaderProgram(const char * vertex_shader_source,
                             const char * fragment_shader_source,
                             const char * geometry_shader_source,
                             const char * tess_control_shader_source,
                             const char * tess_evaluation_shader_source)
{
    this->createFromCStyleString(vertex_shader_source, 
                                 fragment_shader_source, 
                                 geometry_shader_source, 
                                 tess_control_shader_source,
                                 tess_evaluation_shader_source);
}
*/

ShaderProgram::ShaderProgram(ShaderProgram && rhs) noexcept
{
    this->program_ = rhs.program_;
    rhs.program_ = 0;
}

ShaderProgram & ShaderProgram::operator= (ShaderProgram && rhs) noexcept
{
    this->program_ = rhs.program_;
    rhs.program_ = 0;
    return *this;
}

ShaderProgram::~ShaderProgram()
{
    this->destory();
}

void ShaderProgram::createFromCStyleString(const char * vertex_shader_source,
                                           const char * fragment_shader_source,
                                           const char * geometry_shader_source,
                                           const char * tess_control_shader_source,
                                           const char * tess_evaluation_shader_source)
{
    //destroy before create
    this->destory();

    //create shader program
    this->program_ = glCreateProgram();


    GLuint shaders[5] = { 0, 0, 0, 0, 0 };

    GLuint types[5] = {
                        GL_VERTEX_SHADER,
                        GL_FRAGMENT_SHADER,
                        GL_GEOMETRY_SHADER,
						GL_GEOMETRY_SHADER,
						GL_GEOMETRY_SHADER
                      };

    const char * sources[5] = {
                                vertex_shader_source,
                                fragment_shader_source,
                                geometry_shader_source,
                                tess_control_shader_source,
                                tess_evaluation_shader_source
                              };

    for (int i = 0; i < 5; ++i)
    {
        if (sources[i] != NULL)
        {
            shaders[i] = glCreateShader(types[i]);
            glShaderSource(shaders[i], 1, &sources[i], 0);
            glCompileShader(shaders[i]);

            //output error message if fails
            glslPrintShaderLog(shaders[i]);

            glAttachShader(program_, shaders[i]);
        }
    }

    glLinkProgram(program_);

    // check if program linked
    GLint success = 0;
    glGetProgramiv(this->program_, GL_LINK_STATUS, &success);
    if (!success)
    {
        char temp[256];
        glGetProgramInfoLog(this->program_, 256, 0, temp);

        std::cerr << "Failed to link program:" << std::endl;
        std::cerr << temp << std::endl;

        glDeleteProgram(this->program_);
        program_ = 0;

        std::exit(EXIT_FAILURE);
    }

    for (int i = 0; i < 5; ++i)
        glDeleteShader(shaders[i]);

}

void ShaderProgram::createFromFile(const std::string & vertex_shader_file,
                                   const std::string & fragment_shader_file,
                                   const std::string & geometry_shader_file,
                                   const std::string & tess_control_shader_file,
                                   const std::string & tess_evaluation_shader_file)
{
    std::string shader_files[5] = { vertex_shader_file, fragment_shader_file, geometry_shader_file, tess_control_shader_file, tess_evaluation_shader_file };
    std::string shader_strs[5];

    for(int i = 0; i < 5; ++i)
    {
        if (shader_files[i].empty() == true)
            continue;

        std::ifstream input_file(shader_files[i]);

        if(input_file.fail() == true)
        {
            std::cerr << "error: can't open file " << shader_files[i] << std::endl;
            std::exit(EXIT_FAILURE);
        }

        std::stringstream sstream;
        sstream << input_file.rdbuf();

        input_file.close();

        shader_strs[i] = sstream.str();

    }

    this->createFromCStyleString(shader_strs[0].empty() ? nullptr : shader_strs[0].c_str(),
                                 shader_strs[1].empty() ? nullptr : shader_strs[1].c_str(),
                                 shader_strs[2].empty() ? nullptr : shader_strs[2].c_str(),
                                 shader_strs[3].empty() ? nullptr : shader_strs[3].c_str(),
                                 shader_strs[4].empty() ? nullptr : shader_strs[4].c_str());

}


void ShaderProgram::createFromString(const std::string & vertex_shader_str,
                                     const std::string & fragment_shader_str,
                                     const std::string & geometry_shader_str,
                                     const std::string & tess_control_shader_str,
                                     const std::string & tess_evaluation_shader_str)
{

    this->createFromCStyleString(vertex_shader_str.empty() ?          nullptr : vertex_shader_str.c_str(),
                                 fragment_shader_str.empty() ?        nullptr : fragment_shader_str.c_str(),
                                 geometry_shader_str.empty() ?        nullptr : geometry_shader_str.c_str(),
                                 tess_control_shader_str.empty() ?    nullptr : tess_control_shader_str.c_str(),
                                 tess_evaluation_shader_str.empty() ? nullptr : tess_evaluation_shader_str.c_str());
}


void ShaderProgram::destory()
{
    glVerify(glDeleteProgram(this->program_));
    this->program_ = 0;
}

void ShaderProgram::enable() const
{
    if (this->isValid())
    {
        glVerify(glUseProgram(this->program_));
    }
}

void ShaderProgram::disable() const
{
    glVerify(glUseProgram(0));
}

bool ShaderProgram::setBool(const std::string & name, bool val) 
{
    return this->setInt(name, val);
}

bool ShaderProgram::setInt(const std::string & name, int val) 
{
    return openGLSetShaderInt(this->program_, name, val);
}

bool ShaderProgram::setFloat(const std::string & name, float val)
{
    return openGLSetShaderFloat(this->program_, name, val);
}

bool ShaderProgram::setVec2(const std::string & name, const Vec2f & val)
{
    return openGLSetShaderVec2(this->program_, name, val);
}

bool ShaderProgram::setVec2(const std::string & name, const Vec2d & val)
{
    return openGLSetShaderVec2(this->program_, name, val);
}

bool ShaderProgram::setVec2(const std::string & name, const glm::vec2 & val)
{
    return openGLSetShaderVec2(this->program_, name, val);
}

bool ShaderProgram::setVec2(const std::string & name, float x, float y)
{
    return openGLSetShaderVec2(this->program_, name, x, y);
}

bool ShaderProgram::setVec3(const std::string & name, const Vec3f & val)
{
    return openGLSetShaderVec3(this->program_, name, val);
}

bool ShaderProgram::setVec3(const std::string & name, const Vec3d & val)
{
    return openGLSetShaderVec3(this->program_, name, val);
}

bool ShaderProgram::setVec3(const std::string & name, const glm::vec3 & val)
{
    return openGLSetShaderVec3(this->program_, name, val);
}

bool ShaderProgram::setVec3(const std::string & name, float x, float y, float z)
{
    return openGLSetShaderVec3(this->program_, name, x, y, z);
}

bool ShaderProgram::setVec4(const std::string & name, const Vec4f & val)
{
    return openGLSetShaderVec4(this->program_, name, val);
}

bool ShaderProgram::setVec4(const std::string & name, const Vec4d & val)
{
    return openGLSetShaderVec4(this->program_, name, val);
}

bool ShaderProgram::setVec4(const std::string & name, const glm::vec4 & val)
{
    return openGLSetShaderVec4(this->program_, name, val);
}

bool ShaderProgram::setVec4(const std::string & name, float x, float y, float z, float w)
{
    return openGLSetShaderVec4(this->program_, name, x, y, z, w);
}

bool ShaderProgram::setMat2(const std::string & name, const Mat2f & val)
{
    return openGLSetShaderMat2(this->program_, name, val);
}

bool ShaderProgram::setMat2(const std::string & name, const Mat2d & val)
{
    return openGLSetShaderMat2(this->program_, name, val);
}

bool ShaderProgram::setMat2(const std::string & name, const glm::mat2 & val)
{
    return openGLSetShaderMat2(this->program_, name, val);
}

bool ShaderProgram::setMat3(const std::string & name, const Mat3f & val)
{
    return openGLSetShaderMat3(this->program_, name, val);
}

bool ShaderProgram::setMat3(const std::string & name, const Mat3d & val)
{
    return openGLSetShaderMat3(this->program_, name, val);
}

bool ShaderProgram::setMat3(const std::string & name, const glm::mat3 & val)
{
    return openGLSetShaderMat3(this->program_, name, val);
}

bool ShaderProgram::setMat4(const std::string & name, const Mat4f & val)
{
    return openGLSetShaderMat4(this->program_, name, val);
}

bool ShaderProgram::setMat4(const std::string & name, const Mat4d & val)
{
    return openGLSetShaderMat4(this->program_, name, val);
}

bool ShaderProgram::setMat4(const std::string & name, const glm::mat4 & val)
{
    return openGLSetShaderMat4(this->program_, name, val);
}

bool ShaderProgram::isValid() const
{
    return glIsProgram(this->program_);
}

unsigned int ShaderProgram::id() const
{
    return this->program_;
}

}// end of namespace dyno
