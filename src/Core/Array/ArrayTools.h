/**
 * Copyright 2021 Xiaowei He
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
#include "Platform.h"
#include "Array/Array.h"

namespace dyno
{
	template<int N>
	struct PlaceHolder
	{
		char data[N];
	};

	template struct PlaceHolder<1>;
	template struct PlaceHolder<2>;
	template struct PlaceHolder<3>;
	template struct PlaceHolder<4>;
	template struct PlaceHolder<5>;
	template struct PlaceHolder<6>;
	template struct PlaceHolder<7>;
	template struct PlaceHolder<8>;

	template<int N>
	void parallel_allocate_for_list(void* lists, void* elements, size_t ele_size, GArray<int> index);

	template<int N>
	void parallel_init_for_list(void* lists, void* elements, size_t ele_size, GArray<int> index);
}