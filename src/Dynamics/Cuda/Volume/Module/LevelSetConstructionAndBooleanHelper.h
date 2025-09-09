/**
 * Copyright 2025 Lixin Ren
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

//#include "Vector.h"
//#include "DataTypes.h"
//#include "Primitive/Primitive3D.h"

#include "Topology/LevelSet.h"
#include "Topology/TriangleSet.h"
#include "Collision/Distance3D.h"
#include "VolumeMacros.h"
namespace dyno
{
	template<typename TDataType>
	class LevelSetConstructionAndBooleanHelper 
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		LevelSetConstructionAndBooleanHelper() {};

		static void initialFromTriangle(
			std::shared_ptr<TriangleSet<TDataType>> triSet,
			Real dx,
			uint padding,
			DistanceField3D<TDataType>& sdf,
			Coord& origin,
			DArray3D<GridType>& gridType,
			DArray3D<int>& closestTriId);

		static void fastIterative(
			DArray3D<Real>& phi,
			DArray3D<GridType>& gridtype,
			DArray3D<uint>& alpha,
			DArray3D<bool>& outside,
			int interval,
			Real dx,
			bool controlInterval);

		static void initialForBoolean(
			DistanceField3D<TDataType>& inA,
			DistanceField3D<TDataType>& inB,
			DistanceField3D<TDataType>& out,
			DArray3D<GridType>& gridType,
			DArray3D<bool>& outside,
			Real dx,
			uint padding,
			int boolType);
	};
}