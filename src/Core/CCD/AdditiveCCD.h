/**
 * Copyright 2022 Zixuan Lu
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
 * @brief This is an implementation of AdditiveCCD based on peridyno.
 * 
 *	Additive CCD calc maximum TOI such that  d(x) > s * (d(x) - xi) + xi 
 *	to satisfy separation in collision handling board phrase, where d(x) is current primitive distance 
 *  and xi is the thickness. s is a rescalar.
 *		
 * For details, refer to "Li M, Kaufman D M, Jiang C.
	 Codimensional incremental potential contact[J]. arXiv preprint arXiv:2012.04457, 2020." chapter 5.
 *
 */

namespace dyno
{
	template<typename T>
	class AdditiveCCD
	{
	public:

		DYN_FUNC AdditiveCCD(T xi_, T s_, T tc_) 
			: xi(xi_), s(s_), tc(tc_) {
		};
		DYN_FUNC AdditiveCCD() = default;
		//default: this is a normal CCD board phrase as thickness = 0, that return TOI as distance >0; TOI in[0,1]
		//with CA strategy.

		/**
		 * @brief Do a continuous collision detection between a vertex and a triangle
		 *
		 * @param x3 The vertex position at t0
		 * @param y3 The vertex position at t1
		 * @param [x0, x1, x2] The triangle at t0
		 * @param [y0, y1, y2] The triangle at t1
		 * @param time the time of impact, initialize its value to 1 if VertexFaceCCD was for the first time called
		 * @return return ture if collision detected, otherwise return false
		 */
		
		inline DYN_FUNC bool VertexFaceCCD(
			const Vector<T, 3>& x0, const Vector<T, 3>& x1, const Vector<T, 3>& x2, const Vector<T, 3>& x3,
			const Vector<T, 3>& y0, const Vector<T, 3>& y1, const Vector<T, 3>& y2, const Vector<T, 3>& y3,
			T& time, T invL);

		/**
		 * @brief Do a continuous collision detection between two edges
		 *
		 * @tparam T
		 * @param [x0, x1] first edge at t0
		 * @param [y0, y1] first edge at t1
		 * @param [x2, x3] second edge at t0
		 * @param [y2, y3] second edge at t1
		 * @param time time of impact, initialize its value to 1 if EdgeEdgeCCD was for the first time called
		 * @return return ture if collision detected, otherwise return false
		 */
		

		inline DYN_FUNC bool EdgeEdgeCCD(
			const Vector<T, 3>& x0, const Vector<T, 3>& x1, const Vector<T, 3>& x2, const Vector<T, 3>& x3,
			const Vector<T, 3>& y0, const Vector<T, 3>& y1, const Vector<T, 3>& y2, const Vector<T, 3>& y3,
			T& time, T invL);


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

		inline DYN_FUNC bool TriangleCCD(
			TTriangle3D<Real>& s0, TTriangle3D<Real>& s1,
			TTriangle3D<Real>& t0, TTriangle3D<Real>& t1,
			Real& toi);

		/**
		 * @brief find the close point between two triangles, store their barycentric coordinates ordered as vertex.
		 *
		 * @tparam T
		 * @param s first triangle 
		 * @param t second triangle 
		 */
		inline DYN_FUNC void projectClosePoint(
			const TTriangle3D<Real>& s, const TTriangle3D<Real>& t,
			Vector<T,3> & first, Vector<T,3> &second);

	private: 


		inline Vector<T, 3> DistanceEE(
			const Vector<T, 3>& x0, const Vector<T, 3>& x1,
			const Vector<T, 3>& y0, const Vector<T, 3>& y1,
			T* para);

		inline T DistanceVF(
			const Vector<T, 3>& x,
			const Vector<T, 3>& y0,
			const Vector<T, 3>& y1,
			const Vector<T, 3>& y2);

		inline Vector<T, 3> DistanceVF_v(
			const Vector<T, 3>& x,
			const Vector<T, 3>& y0,
			const Vector<T, 3>& y1,
			const Vector<T, 3>& y2,
			T* para);

		T SquareDistanceVF(const Vector<T, 3>& x0, const Vector<T, 3>& x1, const Vector<T, 3>& x2,
			const Vector<T, 3>& x3);

		T SquareDistanceEE(
			const Vector<T, 3>& x0, const Vector<T, 3>& x1,
			const Vector<T, 3>& x2, const Vector<T, 3>& x3);

		T s = 0.2 ; //separation rescale, usually <<1 in [0,1].
		T xi = 0.0; // thickness required, rescalar to max primitive length (e.g. xi times of max edge of triangle)
		T tc = 0.95; // intuitive time-to-collide lower bound, can simply set as 1.
	};
}

#include "additiveCCD.inl"