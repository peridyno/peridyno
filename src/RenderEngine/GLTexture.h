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
#include <glm/glm.hpp>

namespace dyno {

	class GLTexture : public GLObject
	{
	public:
		virtual void create();
		virtual void release();

		virtual void bind();
		virtual void unbind();

		virtual void bind(int slot);

	public:
		unsigned int target;
		unsigned int internalFormat;
		unsigned int format;
		unsigned int type;

		unsigned int minFilter;
		unsigned int maxFilter;
		unsigned int wrapS;
		unsigned int wrapT;

		glm::vec4	 borderColor;
	};

	class GLTexture2D : public GLTexture
	{
	public:
		GLTexture2D();

		virtual void create();

		virtual void resize(int w, int h);
		virtual void load(int w, int h, void* data);

		// OpenGL 4.4+, clear texture
		virtual void clear(void* value);

		void genMipmap();
	};

	class GLTexture2DArray : public GLTexture
	{
	public:
		GLTexture2DArray();

		virtual void create();
		virtual void resize(int w, int h, int layers);
	};

}

