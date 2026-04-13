/**
 * Copyright 2025 Xiaowei He
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
#include "Vector.h"
#include "Primitive/Primitive3D.h"

namespace dyno
{
	template<typename T>
	class LightCCD
	{
	public:
		/**
		 * @brief Do a continuous collision detection between a vertex and a triangle
		 *
		 * @param p0 The vertex position at t0
		 * @param p1 The vertex position at t1
		 * @param [a0, b0, c0] The triangle at t0
		 * @param [a1, b1, c1] The triangle at t1
		 * @param time the time of impact, initialize its value to 1 if VertexFaceCCD was for the first time called
		 * @return return ture if collision detected, otherwise return false
		 */
		
		static inline DYN_FUNC bool VertexFaceCCD(
			const Vector<T, 3>& p0, const Vector<T, 3>& a0, const Vector<T, 3>& b0, const Vector<T, 3>& c0,
			const Vector<T, 3>& p1, const Vector<T, 3>& a1, const Vector<T, 3>& b1, const Vector<T, 3>& c1,
			T& time, T pid, T tid);
	};
}

#include "LightCCD.inl"