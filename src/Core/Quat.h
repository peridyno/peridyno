/**
 * @file Quat.h
 * @brief Implementation of quaternion
 * 
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
 * 
 */
#pragma once
#include "Vector.h"
#include "Matrix.h"
#define GLM_FORCE_RADIANS

namespace dyno 
{
	/*
	 * Quaternion is defined for float and double, all functions are taking radians as parameters
	 */
	template <typename Real>
	class Quat
	{
	public:
		/* Constructors */
		DYN_FUNC Quat();

		/**
		 * @brief Construct a quaternion, be aware that w is the scalar part and (x, y, z) is the vector part 
		 * 
		 * @return DYN_FUNC 
		 */
		DYN_FUNC Quat(Real x, Real y, Real z, Real w);

		/**
		 * @brief Construct a quaternion from a rotation and a unit vector
		 * 
		 * @param axis should be a unit vector
		 * @param rot rotation in radian
		 * @return DYN_FUNC 
		 */
		DYN_FUNC Quat(Real rot, const Vector<Real, 3> &axis);  //init from the rotation axis and angle(in radian)

		DYN_FUNC Quat(const Quat<Real> &);
		DYN_FUNC explicit Quat(const SquareMatrix<Real, 3> &);   //init from a 3x3matrix
		DYN_FUNC explicit Quat(const SquareMatrix<Real, 4> &);    //init from a 4x4matrix

		// yaw (Z), pitch (Y), roll (X);     
		DYN_FUNC explicit Quat(const Real yaw, const Real pitch, const Real roll);

		/* Assignment operators */
		DYN_FUNC Quat<Real> &operator = (const Quat<Real> &);
		DYN_FUNC Quat<Real> &operator += (const Quat<Real> &);
		DYN_FUNC Quat<Real> &operator -= (const Quat<Real> &);

		/* Special functions */
		DYN_FUNC Real norm() const;
		DYN_FUNC Real normSquared() const;
		DYN_FUNC Quat<Real>& normalize();

		DYN_FUNC Quat<Real> inverse() const;

		DYN_FUNC Real angle() const;                                         // return the angle between this quat and the identity quaternion.
		DYN_FUNC Real angle(const Quat<Real>&) const;						// return the angle between this and the argument
		DYN_FUNC Quat<Real> conjugate() const;									// return the conjugate

		/**
		 * @brief Rotate a vector by the quaternion,
		 *		  guarantee the quaternion is normalized before rotating the vector
		 * 
		 * @return v' where (0, v') is calculate by q(0, v)q^{*}.  
		 */
		DYN_FUNC Vector<Real, 3> rotate(const Vector<Real, 3>& v) const;

		DYN_FUNC void toRotationAxis(Real &rot, Vector<Real, 3> &axis) const;

		DYN_FUNC void toEulerAngle(Real& yaw, Real& pitch, Real& roll) const;

		DYN_FUNC SquareMatrix<Real, 3> toMatrix3x3() const;                    //return 3x3matrix format
		DYN_FUNC SquareMatrix<Real, 4> toMatrix4x4() const;                    //return 4x4matrix with a identity transform.


		/* Operator overloading */
		DYN_FUNC Quat<Real> operator - (const Quat<Real>&) const;
		DYN_FUNC Quat<Real> operator - (void) const;
		DYN_FUNC Quat<Real> operator + (const Quat<Real>&) const;
		DYN_FUNC Quat<Real> operator * (const Quat<Real>&) const;
		DYN_FUNC Quat<Real> operator * (const Real&) const;
		DYN_FUNC Quat<Real> operator / (const Real&) const;
		DYN_FUNC bool operator == (const Quat<Real>&) const;
		DYN_FUNC bool operator != (const Quat<Real>&) const;
		DYN_FUNC Real dot(const Quat<Real> &) const;

		DYN_FUNC static inline Quat<Real> identity() { return Quat<Real>(0, 0, 0, 1); }
		DYN_FUNC static inline Quat<Real> fromEulerAngles(const Real& yaw, const Real& pitch, const Real& roll) {
			Real cr = glm::cos(roll * 0.5);
			Real sr = glm::sin(roll * 0.5);
			Real cp = glm::cos(pitch * 0.5);
			Real sp = glm::sin(pitch * 0.5);
			Real cy = glm::cos(yaw * 0.5);
			Real sy = glm::sin(yaw * 0.5);

			Quat<Real> q;
			q.w = cr * cp * cy + sr * sp * sy;
			q.x = sr * cp * cy - cr * sp * sy;
			q.y = cr * sp * cy + sr * cp * sy;
			q.z = cr * cp * sy - sr * sp * cy;

			return q;
		}

	public:
		Real x, y, z, w;
	};

	//make * operator commutative
	template <typename S, typename T>
	DYN_FUNC inline Quat<T> operator *(S scale, const Quat<T> &quad)
	{
		return quad * scale;
	}

	template class Quat<float>;
	template class Quat<double>;
	//convenient typedefs
	typedef Quat<float> Quat1f;
	typedef Quat<double> Quat1d;
    
}//end of namespace dyno
#include "Quat.inl"
