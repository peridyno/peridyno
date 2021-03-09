/*
* @file shader_program.h
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
#pragma once

#include <glm/fwd.hpp>

#include "Vector.h"
#include "Matrix.h"

namespace dyno {

class ShaderProgram
{
public:

    ShaderProgram() = default;

    //disable copy
    ShaderProgram(const ShaderProgram & rhs) = delete;
    ShaderProgram & operator = (const ShaderProgram & rhs) = delete;

    //enable move
    ShaderProgram(ShaderProgram && rhs) noexcept;
    ShaderProgram & operator = (ShaderProgram && rhs) noexcept;

    ~ShaderProgram();

    /*
    ShaderProgram(const char * vertex_shader_source,
                  const char * fragment_shader_source,
                  const char * geometry_shader_source = nullptr,
                  const char * tess_control_shader_source = nullptr,
                  const char * tess_evaluation_shader_source = nullptr);
    */

    void createFromCStyleString(const char * vertex_shader_source,
                                const char * fragment_shader_source,
                                const char * geometry_shader_source = nullptr,
                                const char * tess_control_shader_source = nullptr,
                                const char * tess_evaluation_shader_source = nullptr);

    void  createFromFile(const std::string & vertex_shader_file,
                         const std::string & fragment_shader_file,
                         const std::string & geometry_shader_file = {},
                         const std::string & tess_control_shader_file = {},
                         const std::string & tess_evaluation_shader_file = {});

    void createFromString(const std::string & vertex_shader_str,
                          const std::string & fragment_shader_str,
                          const std::string & geometry_shader_str = {},
                          const std::string & tess_control_shader_str = {},
                          const std::string & tess_evaluation_shader_str = {});

    void destory();

    void enable() const;
    void disable() const;

    //setter
    bool setBool(const std::string & name, bool val);
    bool setInt(const std::string & name, int val);
    bool setFloat(const std::string & name, float val);

    bool setVec2(const std::string & name, const Vector2f & val);
    bool setVec2(const std::string & name, const Vector2d & val); //degrade to float type
    bool setVec2(const std::string & name, const glm::vec2 & val);
    bool setVec2(const std::string & name, float x, float y);

    bool setVec3(const std::string & name, const Vector3f & val);
    bool setVec3(const std::string & name, const Vector3d & val); //degrade to float type
    bool setVec3(const std::string & name, const glm::vec3 & val);
    bool setVec3(const std::string & name, float x, float y, float z);
    
    bool setVec4(const std::string & name, const Vector4f & val);
    bool setVec4(const std::string & name, const Vector4d & val); //degrade to float type
    bool setVec4(const std::string & name, const glm::vec4 & val);
    bool setVec4(const std::string & name, float x, float y, float z, float w);

    bool setMat2(const std::string & name, const Matrix2f & val);
    bool setMat2(const std::string & name, const Matrix2d & val); //degrade to float type
    bool setMat2(const std::string & name, const glm::mat2 & val);

    bool setMat3(const std::string & name, const Matrix3f & val);
    bool setMat3(const std::string & name, const Matrix3d & val); //degrade to float type
    bool setMat3(const std::string & name, const glm::mat3 & val);
    
    bool setMat4(const std::string & name, const Matrix4f & val);
    bool setMat4(const std::string & name, const Matrix4d & val); //degrade to float type
    bool setMat4(const std::string & name, const glm::mat4 & val);

    bool isValid() const;
    unsigned int id() const;

private:
    unsigned int program_ = 0;
};

} // end of namespace dyno
