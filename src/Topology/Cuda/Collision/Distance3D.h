/**
 * Copyright 2024 Xiaowei He
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
#include "Matrix.h"
#include "Quat.h"

#include "Primitive/Primitive3D.h"

#include "Module/TopologyModule.h"
#include "Topology/EdgeSet.h"

namespace dyno
{
	template<typename Real>
	struct ProjectedPoint3D
	{
	public:
		typedef  Vector<Real, 2> Coord2D;
		typedef  Vector<Real, 3> Coord3D;

	public:

		int id = -1;   //used to tag the primitive id
		Real signed_distance; //signed distance

		Coord3D point;
		Coord3D normal;
	};

	/**
	 * @brief Calculate the signed distance from a point to a triangular mesh
	 * 
	 * @param p3d return the cloest point on the triangular mesh
	 * @param point input point
	 * @param vertices vertices of the triangular mesh
	 * @param indices indices of the triangular mesh
	 * @param list tiangule ids to be checked
	 * @param dHat the thickness of the triangular mesh
	 * @return return false when the input list is empty, in this case do not use p3d in the following code
	 */
	template<typename Real, typename Coord, DeviceType deviceType, typename IndexType>
	DYN_FUNC bool calculateSignedDistance2TriangleSet(ProjectedPoint3D<Real>& p3d, Coord point, Array<Coord, deviceType>& vertices, Array<TopologyModule::Triangle, deviceType>& indices, List<IndexType>& list, Real dHat = 0);

	template<typename Real, typename Coord, DeviceType deviceType, typename IndexType>
	DYN_FUNC bool calculateSignedDistance2TriangleSetFromNormal(
		ProjectedPoint3D<Real>& p3d,
		Coord point,
		Array<Coord, deviceType>& vertices,
		Array<TopologyModule::Edge, deviceType>& edges,
		Array<TopologyModule::Triangle, deviceType>& triangles,
		Array<TopologyModule::Tri2Edg, deviceType>& t2e,
		Array<Coord, deviceType>& edgeNormal,
		Array<Coord, deviceType>& vertexNormal,
		List<IndexType>& list,
		Real dHat = 0);


	/**
	 * @brief Calculate the distance from a point to a triangular mesh
	 *
	 * @param p3d return the cloest point on the triangular mesh
	 * @param point input point
	 * @param vertices vertices of the triangular mesh
	 * @param indices indices of the triangular mesh
	 * @param list tiangule ids to be checked
	 * @param dHat the thickness of the triangular mesh
	 * @return return false when the input list is empty, in this case do not use p3d in the following code
	 */
	template<typename Real, typename Coord, DeviceType deviceType, typename IndexType>
	DYN_FUNC bool calculateDistance2TriangleSet(ProjectedPoint3D<Real>& p3d, Coord point, Array<Coord, deviceType>& vertices, Array<TopologyModule::Triangle, deviceType>& indices, List<IndexType>& list, Real dHat = 0);

}

#include "Distance3D.inl"
