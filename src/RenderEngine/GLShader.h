/**
 * Copyright 2017-2021 Jian SHI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "GLObject.h"
#include <string>
#include "Vector.h"
#include <glm/glm.hpp>

namespace dyno {

	class GLShader : public GLObject
	{
	public:
		GLShader() {}
		bool createFromFile(unsigned int type, const std::string& path);
		bool createFromSource(unsigned int type, const std::string& src);
		void release();

	protected:
		void create() {};

	};

	class GLShaderProgram : public GLObject
	{
	public:
		void create();
		void release();

		void attachShader(const GLShader& shader);
		bool link();

		void use();

		//
		void setFloat(const char* name, float v);
		void setInt(const char* name, int v);
		void setVec4(const char* name, Vec4f v);
		void setVec3(const char* name, Vec3f v);
		void setVec2(const char* name, Vec2f v);
	};

	// public helpe function...
	GLShaderProgram CreateShaderProgram(const char* vs, const char* fs, const char* gs = 0);

}