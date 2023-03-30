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
#include "Vector.h"
#include "Primitive/Primitive3D.h"
/**
 * @brief This is an implementation of TightCCD based on peridyno.
 * 
 * For details, refer to "TightCCD: Efficient and Robust Continuous Collision Detection using Tight Error Bounds"
 *		by Wang et al., Pacific Graphics 2015 
 * 
 * Be aware TightCCD can remove all the false negatives but not all the false positives 
 *		refer to "A Large Scale Benchmark and an Inclusion-Based Algorithm for Continuous Collision Detection", TOG 2020,
*		for a comparison for advantages and disadvantages of all CCD methods
 */

namespace dyno
{
	template<typename T>
	class TightCCD
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
			T& time);

		/**
		 * @brief Do a continuous collision detection between two edges
		 *
		 * @tparam T
		 * @param [a0, b0] first edge at t0
		 * @param [a1, b1] first edge at t1
		 * @param [c0, d0] second edge at t0
		 * @param [a1, b1] second edge at t1
		 * @param time time of impact, initialize its value to 1 if EdgeEdgeCCD was for the first time called
		 * @return return ture if collision detected, otherwise return false
		 */
		static inline DYN_FUNC bool EdgeEdgeCCD(
			const Vector<T, 3>& a0, const Vector<T, 3>& b0, const Vector<T, 3>& c0, const Vector<T, 3>& d0,
			const Vector<T, 3>& a1, const Vector<T, 3>& b1, const Vector<T, 3>& c1, const Vector<T, 3>& d1,
			T& time);


		/**
		 * @brief Do a continuous collision detection between two triangles
		 *
		 * @tparam T
		 * @param s0 first triangle at t0
		 * @param s1 first triangle at t1
		 * @param t0 second triangle at t0
		 * @param t1 second triangle at t1
		 * @param time time of impact, initialize its value to 1 if TriangleCCD was for the first time called
		 * @return return ture if collision detected, otherwise return false
		 */
		static inline DYN_FUNC bool TriangleCCD(
			TTriangle3D<Real>& s0, TTriangle3D<Real>& s1,
			TTriangle3D<Real>& t0, TTriangle3D<Real>& t1,
			Real& toi);
	};
}

#include "TightCCD.inl"