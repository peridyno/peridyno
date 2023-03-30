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

#include <typeinfo>
#include <iostream>

namespace gl
{
	/*
	 * Base OpenGL object with ID and create/release interface
	 */
	class Object
	{
	protected:
		virtual void create() = 0;
		virtual void release() = 0;

		virtual bool isValid() const { return id != 0xFFFFFFFF; }

	protected:
		Object() = default;
		virtual ~Object() = default; 
		
		// should be non-copyable
		Object(const Object&) = delete;
		Object& operator = (const Object&) = delete;

	public:
		unsigned int id = 0xFFFFFFFF;	// GL_INVALID_INDEX
	};

	// helper functions
	unsigned int glCheckError_(const char* file, int line);
	#define glCheckError() glCheckError_(__FILE__, __LINE__) 

#define GL_OBJECT(T) \
	public:	~T() { if(isValid()) printf("Unreleased resource: %s(%d)\n", #T, id);}
}