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

namespace gl
{
	class Buffer : public Object
	{
	public:
		virtual void create(int target, int usage);
		virtual void release();

		void bind();
		void unbind();

		virtual void allocate(int size);
		virtual void load(void* data, int size, int offset = 0);

		// for uniform buffer
		void bindBufferBase(int idx);

	private:
		virtual void create();

	protected:
		int target = -1;
		int usage = -1;
		int size = -1;
	};	   
}