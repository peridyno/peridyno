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

#include "Object.h"
#include <string>
#include "Vector.h"
#include <glm/glm.hpp>


namespace gl {

	// OpenGL shader
	class Shader : public Object
	{
		GL_OBJECT(Shader)
	public:
		bool createFromFile(unsigned int type, const std::string& path);
		bool createFromSource(unsigned int type, const std::string& src);
		bool createFromSPIRV(unsigned int type, const void* spirv, const size_t len);
		void release();

	protected:
		void create() {};
	};

	// OpenGL shader program
	class Program : public Object
	{
		GL_OBJECT(Program)
	public:
		void create();
		void release();

		void attachShader(const Shader& shader);
		bool link();

		void use();

		//
		void setFloat(const char* name, float v);
		void setInt(const char* name, int v);

		void setVec4(const char* name, dyno::Vec4f v);
		void setVec3(const char* name, dyno::Vec3f v);
		void setVec2(const char* name, dyno::Vec2f v);

	public:
		static Program* createProgram(const char* vs, const char* fs, const char* gs = 0);

		static Program* createProgramSPIRV(
			const void* vs, size_t vs_len,
			const void* fs, size_t fs_len,
			const void* gs = 0, size_t gs_len = 0);
	};

}