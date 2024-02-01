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
#include "EdgeSet.h"
#include "TriangleSet.h"
#include "TetrahedronSet.h"

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

		void setEdgeIndex(const DArray<Edge>& segments) { mEdgeIndex.assign(segments); }
		void setEdgeIndex(const CArray<Edge>& segments) { mEdgeIndex.assign(segments); }

		void setTriangleIndex(const DArray<Triangle>& triangles) { mTriangleIndex.assign(triangles); }
		void setTriangleIndex(const CArray<Triangle>& triangles) { mTriangleIndex.assign(triangles); }

		void setTetrahedronIndex(const DArray<Tetrahedron>& tetrahedrons) { mTetrahedronIndex.assign(tetrahedrons); }
		void setTetrahedronIndex(const CArray<Tetrahedron>& tetrahedrons) { mTetrahedronIndex.assign(tetrahedrons); }

		/**
		 * @brief return the 1-dimensional simplex to EdgeSet
		 */
		void extractSimplex1D(EdgeSet<TDataType>& es);

		/**
		 * @brief return the 2-dimensional simplex to TriangleSet
		 */
		void extractSimplex2D(TriangleSet<TDataType>& ts);

		/**
		 * @brief return the 3-dimensional simplex to TetrahedronSet
		 */
		void extractSimplex3D(TetrahedronSet<TDataType>& ts);

		void extractPointSet(PointSet<TDataType>& ps);

		/**
		 * @brief extract and merge edges from 1, 2, 3-dimensional simplexes into one EdgeSet
		 */
		void extractEdgeSet(EdgeSet<TDataType>& es);

		/**
		 * @brief extract and merge triangles from 2, 3-dimensional simplexes into one TriangleSet
		 */
		void extractTriangleSet(TriangleSet<TDataType>& ts);

	protected:
		void updateTopology() override;

	private:
		DArray<Edge> mEdgeIndex;
		DArray<Triangle> mTriangleIndex;
		DArray<Tetrahedron> mTetrahedronIndex;
	};
}

