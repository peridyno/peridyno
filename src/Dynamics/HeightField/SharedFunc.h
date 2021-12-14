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
#include "Array/Array.h"
#include "Array/ArrayList.h"

#include "Peridynamics/NeighborData.h"
#include "DataTypes.h"


#include "vector_types.h"
#include <string>
#include <assert.h>

namespace dyno {
	template<typename Coord, typename NPair>
	void constructRestShape(
		DArrayList<NPair>& shape,
		DArrayList<int>& nbr,
		DArray<Coord>& pos);

	typedef uchar4 rgb;

	typedef float4 vertex; // x, h, z

	typedef float4 gridpoint; // h, uh, vh, b

	typedef int reflection;

	#define M_PI 3.14159265358979323846
	#define M_E 2.71828182845904523536
	#define EPSILON 0.000001

	#define cudaCheck(x) { cudaError_t err = x;  }
	#define synchronCheck {}
}
