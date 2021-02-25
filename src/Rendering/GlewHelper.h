/*
 * @file glew_utilities.h 
 * @Brief openGL glew utilities
 * @author: Wei Chen
 * 
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

/* Note: since glew.h must be included before gl.h to pass compiler,
 *       you need include this file (glew_utilities.h) before opengl_primitives.h
 *       
 *       This file is used to add utility function that use glew library
 */

#pragma  once

#include <iostream>

#include "gl_utilities.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Vector.h"
#include "Matrix.h"

namespace dyno{
inline int openGLGetCurBindShaderID()
{
    GLint cur_program_id = 0;
    glVerify(glGetIntegerv(GL_CURRENT_PROGRAM, &cur_program_id));
    return cur_program_id;
}

inline int openGLGetCurShaderAttribLocation(const std::string & name)
{
    int cur_shader_id = openGLGetCurBindShaderID();
    return glGetAttribLocation(cur_shader_id, name.c_str());
}

inline void openGLCheckShaderBind(int shader_id)
{
    GLint cur_program_id = 0;
    glVerify(glGetIntegerv(GL_CURRENT_PROGRAM, &cur_program_id));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define DEBUG_OUTPUT false

inline bool openGLSetShaderInt(int shader_id, const std::string & name, int val)
{
    if(shader_id == 0)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: cur bind shader is 0! " << std::endl;
        #endif
        return false;
    }
    openGLCheckShaderBind(shader_id);

    int location = glGetUniformLocation(shader_id, name.c_str());
    if (location == -1)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: no uniform attribute " << name << " found!" << std::endl;
        #endif
        return false;
    }

    glVerify(glUniform1i(location, val));
    return true;
}

inline bool openGLSetShaderBool(int shader_id, const std::string & name, bool val)
{
    return openGLSetShaderInt(shader_id, name, val);
}

inline bool openGLSetShaderFloat(int shader_id, const std::string & name, float val)
{
    if(shader_id == 0)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: cur bind shader is 0! " << std::endl;
        #endif
        return false;
    }
    openGLCheckShaderBind(shader_id);

    int location = glGetUniformLocation(shader_id, name.c_str());
    if (location == -1)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: no uniform attribute " << name << " found!" << std::endl;
        #endif
        return false;
    }

    glVerify(glUniform1f(location, val));
    return true;
}

inline bool openGLSetShaderFloat2V(int shader_id, const std::string & name, unsigned int num, const float * val)
{
    if(shader_id == 0)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: cur bind shader is 0! " << std::endl;
        #endif
        return false;
    }
    openGLCheckShaderBind(shader_id);

    int location = glGetUniformLocation(shader_id, name.c_str());
    if (location == -1)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: no uniform attribute " << name << " found!" << std::endl;
        #endif
        return false;
    }

    glVerify(glUniform2fv(location, num, val));
    return true;
}

inline bool openGLSetShaderVec2(int shader_id, const std::string & name, const glm::vec2 & val)
{
    if(shader_id == 0)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: cur bind shader is 0! " << std::endl;
        #endif
        return false;
    }
    openGLCheckShaderBind(shader_id);

    int location = glGetUniformLocation(shader_id, name.c_str());
    if (location == -1)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: no uniform attribute " << name << " found!" << std::endl;
        #endif
        return false;
    }

    glVerify(glUniform2fv(location, 1, glm::value_ptr(val)));
    return true;
}

template <typename T>
inline bool openGLSetShaderVec2(int shader_id, const std::string & name, const Vector<T, 2> & val)
{
    glm::vec2 glm_val = { val[0], val[1] };
    return openGLSetShaderVec2(shader_id, name, glm_val);
}

inline bool openGLSetShaderVec2(int shader_id, const std::string & name, float x, float y)
{
    if(shader_id == 0)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: cur bind shader is 0! " << std::endl;
        #endif
        return false;
    }
    openGLCheckShaderBind(shader_id);

    int location = glGetUniformLocation(shader_id, name.c_str());
    if (location == -1)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: no uniform attribute " << name << " found!" << std::endl;
        #endif
        return false;
    }

    glVerify(glUniform2f(location, x, y));
    return true;
}

inline bool openGLSetShaderVec3(int shader_id, const std::string & name, const glm::vec3 & val)
{
    if(shader_id == 0)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: cur bind shader is 0! " << std::endl;
        #endif
        return false;
    }
    openGLCheckShaderBind(shader_id);

    int location = glGetUniformLocation(shader_id, name.c_str());
    if (location == -1)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: no uniform attribute " << name << " found!" << std::endl;
        #endif
        return false;
    }

    glVerify(glUniform3fv(location, 1, glm::value_ptr(val)));
    return true;
}

template <typename T>
inline bool openGLSetShaderVec3(int shader_id, const std::string & name, const Vector<T, 3> & val)
{
    glm::vec3 glm_val = { val[0], val[1], val[2] };
    return openGLSetShaderVec3(shader_id, name, glm_val);
}


inline bool openGLSetShaderVec3(int shader_id, const std::string & name, float x, float y, float z)
{
    if (shader_id == 0)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: cur bind shader is 0! " << std::endl;
        #endif
        return false;
    }
    openGLCheckShaderBind(shader_id);

    int location = glGetUniformLocation(shader_id, name.c_str());
    if (location == -1)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: no uniform attribute " << name << " found!" << std::endl;
        #endif
        return false;
    }

    glVerify(glUniform3f(location, x, y, z));
    return true;
}

inline bool openGLSetShaderVec4(int shader_id, const std::string & name, const glm::vec4 & val)
{
    if (shader_id == 0)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: cur bind shader is 0! " << std::endl;
        #endif
        return false;
    }
    openGLCheckShaderBind(shader_id);

    int location = glGetUniformLocation(shader_id, name.c_str());
    if (location == -1)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: no uniform attribute " << name << " found!" << std::endl;
        #endif
        return false;
    }

    glVerify(glUniform4fv(location, 1, glm::value_ptr(val)));
    return true;
}

template <typename T>
inline bool openGLSetShaderVec4(int shader_id, const std::string & name, const Vector<T, 4> & val)
{
    glm::vec4 glm_val = { val[0], val[1], val[2], val[3] };
    return openGLSetShaderVec4(shader_id, name, glm_val);
}


inline bool openGLSetShaderVec4(int shader_id, const std::string & name, float x, float y, float z, float w)
{
    if (shader_id == 0)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: cur bind shader is 0! " << std::endl;
        #endif
        return false;
    }
    openGLCheckShaderBind(shader_id);

    int location = glGetUniformLocation(shader_id, name.c_str());
    if (location == -1)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: no uniform attribute " << name << " found!" << std::endl;
        #endif
        return false;
    }

    glVerify(glUniform4f(location, x, y, z, w));
    return true;
}

inline bool openGLSetShaderFloat4V(int shader_id, const std::string & name, unsigned int num, const float * val)
{
    if (shader_id == 0)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: cur bind shader is 0! " << std::endl;
        #endif
        return false;
    }
    openGLCheckShaderBind(shader_id);

    int location = glGetUniformLocation(shader_id, name.c_str());
    if (location == -1)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: no uniform attribute " << name << " found!" << std::endl;
        #endif
        return false;
    }

    glVerify(glUniform4fv(location, num, val));
    return true;
}

inline bool openGLSetShaderMat2(int shader_id, const std::string & name, const glm::mat2 & val)
{
    if (shader_id == 0)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: cur bind shader is 0! " << std::endl;
        #endif
        return false;
    }
    openGLCheckShaderBind(shader_id);

    int location = glGetUniformLocation(shader_id, name.c_str());
    if (location == -1)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: no uniform attribute " << name << " found!" << std::endl;
        #endif
        return false;
    }

    glVerify(glUniformMatrix2fv(location, 1, GL_FALSE, glm::value_ptr(val)));
    return true;
}

template <typename T>
inline bool openGLSetShaderMat2(int shader_id, const std::string & name, const SquareMatrix<T, 2> & val)
{
    glm::mat2 glm_mat = {val(0,0), val(1,0),  //col 0
                         val(0,1), val(1,1)}; //col 1
    return openGLSetShaderMat2(shader_id, name, glm_mat);
}

inline bool openGLSetShaderMat3(int shader_id, const std::string & name, const glm::mat3 & val)
{
    if (shader_id == 0)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: cur bind shader is 0! " << std::endl;
        #endif
        return false;
    }
    openGLCheckShaderBind(shader_id);

    int location = glGetUniformLocation(shader_id, name.c_str());
    if (location == -1)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: no uniform attribute " << name << " found!" << std::endl;
        #endif
        return false;
    }

    glVerify(glUniformMatrix3fv(location, 1, GL_FALSE, glm::value_ptr(val)));
    return true;
}

template <typename T>
inline bool openGLSetShaderMat3(int shader_id, const std::string & name, const SquareMatrix<T, 3> & val)
{
    glm::mat3 glm_mat = { val(0,0), val(1,0), val(2,0),    //col 0
                          val(0,1), val(1,1), val(2,1),    //col 1
                          val(0,2), val(1,2), val(2,2)};   //col 2
    return openGLSetShaderMat3(shader_id, name, glm_mat);
}

inline bool openGLSetShaderMat4(int shader_id, const std::string & name, const glm::mat4 & val)
{
    if(shader_id == 0)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: cur bind shader is 0! " << std::endl;
        #endif
        return false;
    }
    openGLCheckShaderBind(shader_id);

    int location = glGetUniformLocation(shader_id, name.c_str());
    if (location == -1)
    {
        #if DEBUG_OUTPUT
        std::cout << "wanning: no uniform attribute " << name << " found!" << std::endl;
        #endif
        return false;
    }

    glVerify(glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(val)));
    return true;
}

template <typename T>
inline bool openGLSetShaderMat4(int shader_id, const std::string & name, const SquareMatrix<T, 4> & val)
{
    glm::mat4 glm_mat = { val(0,0), val(1,0), val(2,0), val(3,0),   //col 0
                          val(0,1), val(1,1), val(2,1), val(3,1),   //col 1
                          val(0,2), val(1,2), val(2,2), val(3,2),   //col 2
                          val(0,3), val(1,3), val(2,3), val(3,3)};  //col 3
    return openGLSetShaderMat4(shader_id, name, glm_mat);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline bool openGLSetCurBindShaderBool(const std::string & name, bool val)
{
    int cur_shader_id = openGLGetCurBindShaderID();
    return openGLSetShaderBool(cur_shader_id, name, val);
}

inline bool openGLSetCurBindShaderInt(const std::string & name, int val)
{
    int cur_shader_id = openGLGetCurBindShaderID();
    return openGLSetShaderInt(cur_shader_id, name, val);
}

inline bool openGLSetCurBindShaderFloat(const std::string & name, float val)
{
    int cur_shader_id = openGLGetCurBindShaderID();
    return openGLSetShaderFloat(cur_shader_id, name, val);
}

inline bool openGLSetCurBindShaderFloat2V(const std::string & name, unsigned int num, const float * val)
{
    int cur_shader_id = openGLGetCurBindShaderID();
    return openGLSetShaderFloat2V(cur_shader_id, name, num, val);
}

inline bool openGLSetCurBindShaderVec2(const std::string & name, const glm::vec2 & val)
{
    int cur_shader_id = openGLGetCurBindShaderID();
    return openGLSetShaderVec2(cur_shader_id, name, val);
}

template <typename T>
inline bool openGLSetCurBindShaderVec2(const std::string & name, const Vector<T, 2> & val)
{
    int cur_shader_id = openGLGetCurBindShaderID();
    return openGLSetShaderVec2(cur_shader_id, name, val);
}

inline bool openGLSetCurBindShaderVec2(const std::string & name, float x, float y)
{
    int cur_shader_id = openGLGetCurBindShaderID();
    return openGLSetShaderVec2(cur_shader_id, name, x, y);
}

inline bool openGLSetCurBindShaderVec3(const std::string & name, const glm::vec3 & val)
{
    int cur_shader_id = openGLGetCurBindShaderID();
    return openGLSetShaderVec3(cur_shader_id, name, val);
}

template <typename T>
inline bool openGLSetCurBindShaderVec3(const std::string & name, const Vector<T, 3> & val)
{
    int cur_shader_id = openGLGetCurBindShaderID();
    return openGLSetShaderVec3(cur_shader_id, name, val);
}

inline bool openGLSetCurBindShaderVec3(const std::string & name, float x, float y, float z)
{
    int cur_shader_id = openGLGetCurBindShaderID();
    return openGLSetShaderVec3(cur_shader_id, name, x, y, z);
}

inline bool openGLSetCurBindShaderVec4(const std::string & name, const glm::vec4 & val)
{
    int cur_shader_id = openGLGetCurBindShaderID();
    return openGLSetShaderVec4(cur_shader_id, name, val);
}

template <typename T>
inline bool openGLSetCurBindShaderVec4(const std::string & name, const Vector<T, 4> & val)
{
    int cur_shader_id = openGLGetCurBindShaderID();
    return openGLSetShaderVec4(cur_shader_id, name, val);
}

inline bool openGLSetCurBindShaderVec4(const std::string & name, float x, float y, float z, float w)
{
    int cur_shader_id = openGLGetCurBindShaderID();
    return openGLSetShaderVec4(cur_shader_id, name, x, y, z, w);
}

inline bool openGLSetCurBindShaderFloat4V( const std::string & name, unsigned int num, const float * val)
{
    int cur_shader_id = openGLGetCurBindShaderID();
    return openGLSetShaderFloat4V(cur_shader_id, name, num, val);
}

inline bool openGLSetCurBindShaderMat2(const std::string & name, const glm::mat2 & val)
{
    int cur_shader_id = openGLGetCurBindShaderID();
    return openGLSetShaderMat2(cur_shader_id, name, val);
}

template <typename T>
inline bool openGLSetCurBindShaderMat2(const std::string & name, const SquareMatrix<T, 2> & val)
{
    int cur_shader_id = openGLGetCurBindShaderID();
    return openGLSetShaderMat2(cur_shader_id, name, val);
}

inline bool openGLSetCurBindShaderMat3(const std::string & name, const glm::mat3 & val)
{
    int cur_shader_id = openGLGetCurBindShaderID();
    return openGLSetShaderMat3(cur_shader_id, name, val);
}

template <typename T>
inline bool openGLSetCurBindShaderMat3(const std::string & name, const SquareMatrix<T, 3> & val)
{
    int cur_shader_id = openGLGetCurBindShaderID();
    return openGLSetShaderMat3(cur_shader_id, name, val);
}

inline bool openGLSetCurBindShaderMat4(const std::string & name, const glm::mat4 & val)
{
    int cur_shader_id = openGLGetCurBindShaderID();
    return openGLSetShaderMat4(cur_shader_id, name, val);
}

template <typename T>
inline bool openGLSetCurBindShaderMat4(const std::string & name, const SquareMatrix<T, 4> & val)
{
    int cur_shader_id = openGLGetCurBindShaderID();
    return openGLSetShaderMat4(cur_shader_id, name, val);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline GLenum openGLCheckCurFramebufferStatus(GLenum target = GL_FRAMEBUFFER)
{
    GLenum status = glCheckFramebufferStatus(target);
    switch (status)
    {
    case GL_FRAMEBUFFER_COMPLETE:
        break;

    case GL_FRAMEBUFFER_UNDEFINED:
    {
        std::cerr << "framebuffer error: GL_FRAMEBUFFER_UNDEFINED " << std::endl;
        break;
    }

    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
    {
        std::cerr << "framebuffer error: GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT " << std::endl;
        break;
    }

    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
    {
        std::cerr << "framebuffer error: GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT " << std::endl;
        break;
    }

    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
    {
        std::cerr << "framebuffer error: GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER" << std::endl;
        break;
    }

    case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
    {
        std::cerr << "framebuffer error: GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER" << std::endl;
        break;
    }

    case GL_FRAMEBUFFER_UNSUPPORTED:
    {
        std::cerr << "framebuffer error: GL_FRAMEBUFFER_UNSUPPORTED" << std::endl;
        break;
    }

    case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
    {
        std::cerr << "framebuffer error: GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE" << std::endl;
        break;
    }

    default:
    {
        std::cerr << "framebuffer error: unknown" << std::endl;
        break;
    }

    }

    return status;
}

}//end of namespace dyno
