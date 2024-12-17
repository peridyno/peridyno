/**
 * Copyright 2022 Xiaowei He
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
#include "DataTypes.h"

#include "Module/TopologyModule.h"

#include "Primitive/Primitive3D.h"
#include "Topology/DistanceField3D.h"

namespace dyno
{
	template<typename TDataType>
	class MarchingCubesHelper 
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		MarchingCubesHelper() {};

		static void reconstructSDF(
			DArray3D<Real>& distances,
			Coord origin,
			Real h,
			DistanceField3D<TDataType>& sdf);

		static void countVerticeNumber(
			DArray<int>& num,
			DArray3D<Real>& distances,
			Real isoValue);

		static void constructTriangles(
			DArray<Coord>& vertices,
			DArray<TopologyModule::Triangle>& triangles,
			DArray<int>& vertNum,
			DArray3D<Real>& distances,
			Coord origin,
			Real isoValue,
			Real h);

		static void countVerticeNumberForClipper(
			DArray<int>& num,
			DistanceField3D<TDataType>& sdf,
			TPlane3D<Real> plane);

		static void constructTrianglesForClipper(
			DArray<Real>& field,
			DArray<Coord>& vertices,
			DArray<TopologyModule::Triangle>& triangles,
			DArray<int>& vertNum,
			DistanceField3D<TDataType>& sdf,
			TPlane3D<Real> plane);


		static void countVerticeNumberForOctree(
			DArray<uint>& num,
			DArray<Coord>& vertices,
			DArray<Real>& sdfs,
			Real isoValue);

		static void constructTrianglesForOctree(
			DArray<Coord>& triangleVertices,
			DArray<TopologyModule::Triangle>& triangles,
			DArray<uint>& num,
			DArray<Coord>& cellVertices,
			DArray<Real>& sdfs,
			Real isoValue);

		static void countVerticeNumberForOctreeClipper(
			DArray<uint>& num,
			DArray<Coord>& vertices,
			TPlane3D<Real> plane);

		static void constructTrianglesForOctreeClipper(
			DArray<Real>& vertSDFs,
			DArray<Coord>& triangleVertices,
			DArray<TopologyModule::Triangle>& triangles,
			DArray<uint>& num,
			DArray<Coord>& cellVertices,
			DArray<Real>& sdfs,
			TPlane3D<Real> plane);
	};
}