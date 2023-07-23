/**
 * Copyright 2023 Xiaowei He
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
#include "PointSet.h"

namespace dyno
{
	template<typename TDataType>
	class SimplexSet : public PointSet<TDataType>
	{
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		typedef typename TopologyModule::Edge Edge;
		typedef typename TopologyModule::Triangle Triangle;
		typedef typename TopologyModule::Tetrahedron Tetrahedron;

	public:
		SimplexSet();
		~SimplexSet() override;

		void copyFrom(SimplexSet<TDataType>& simplex);

		bool isEmpty() override;

		void setSegments(const DArray<Edge>& segments) { mSegmentIndex.assign(segments); }
		void setTriangles(const DArray<Triangle>& triangles) { mTriangleIndex.assign(triangles); }
		void setTetrahedrons(const DArray<Tetrahedron>& tetrahedrons) { mTetrahedronSet.assign(tetrahedrons); }

	protected:
		void updateTopology() override;

	private:
		DArray<Edge> mSegmentIndex;
		DArray<Triangle> mTriangleIndex;
		DArray<Tetrahedron> mTetrahedronSet;
	};
}

