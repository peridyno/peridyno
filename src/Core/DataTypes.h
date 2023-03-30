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
#define GLM_FORCE_PURE
#include "Vector.h"
#include "Matrix.h"
#include "Rigid/rigid.h"

namespace dyno
{
	template<class TReal, class TCoord, class TMatrix, class TRigid>
	class DataTypes
	{
	public:
		typedef TReal Real;
		typedef TCoord Coord;
		typedef TMatrix Matrix;
		typedef TRigid Rigid;

		static const char* getName();
	};

	/// 1f DOF, single precision
	typedef DataTypes<float, float, float, Rigid<float, 1>> DataType1f;
	template<> inline const char* DataType1f::getName() { return "DataType1f"; }

	/// 2f DOF, single precision
	typedef DataTypes<float, Vec2f, Mat2f, Rigid2f> DataType2f;
	template<> inline const char* DataType2f::getName() { return "DataType2f"; }

	/// 3f DOF, single precision
	typedef DataTypes<float, Vec3f, Mat3f, Rigid3f> DataType3f;
	template<> inline const char* DataType3f::getName() { return "DataType3f"; }

	/// 1d DOF, double precision
	typedef DataTypes<double, float, float, Rigid<double, 1>> DataType1d;
	template<> inline const char* DataType1d::getName() { return "DataType1d"; }

	/// 2d DOF, double precision
	typedef DataTypes<double, Vec2d, Mat2d, Rigid2d> DataType2d;
	template<> inline const char* DataType2d::getName() { return "DataType2d"; }

	/// 3d DOF, double precision
	typedef DataTypes<double, Vec3d, Mat3d, Rigid3d> DataType3d;
	template<> inline const char* DataType3d::getName() { return "DataType3d"; }
}


