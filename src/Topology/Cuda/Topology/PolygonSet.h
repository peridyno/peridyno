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
#include "PointSet.h"
#include "EdgeSet.h"
#include "TriangleSet.h"
#include "QuadSet.h"

namespace dyno
{
	/**
	 * @brief a class to store polygon whose vertex number is greater or equal than 3
	 */
	template<typename TDataType>
	class PolygonSet : public EdgeSet<TDataType>
	{
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		typedef typename TopologyModule::Edge Edge;
		typedef typename TopologyModule::Triangle Triangle;
		typedef typename TopologyModule::Quad Quad;

	public:
		PolygonSet();
		~PolygonSet() override;

		/**
		 * @brief initialize polygon indices
		 */
		void setPolygons(const CArrayList<uint>& indices);
		void setPolygons(const DArrayList<uint>& indices);

		DArrayList<uint>& polygonIndices() { return mPolygonIndex; }

		const DArrayList<uint>& vertex2Polygon() { return mVer2Poly; }

		const DArrayList<uint>& polygon2Edge() { return mPoly2Edg; };

		const DArray<::dyno::TopologyModule::Edg2Poly>& edge2Polygon() {
			return mEdg2Poly;
		}


		void copyFrom(PolygonSet<TDataType>& polygons);

		bool isEmpty() override;

		/**
		 * @brief extract and merge edges from all polygons into one EdgeSet
		 */
		void extractEdgeSet(EdgeSet<TDataType>& es);

		/**
		 * @brief extract all triangles
		 */
		void extractTriangleSet(TriangleSet<TDataType>& ts);

		/**
		 * @brief extract all quads
		 */
		void extractQuadSet(QuadSet<TDataType>& qs);

		/**
		 * @brief turn all polygons into triangles and store into one TriangleSet
		 */
		void turnIntoTriangleSet(TriangleSet<TDataType>& ts);

		/**
		 * @brief turn all polygons into triangles and store into one TriangleSet
		 */
		void triangleSetToPolygonSet(TriangleSet<TDataType>& ts);

	protected:
		void updateTopology() override;

		void updateEdges() override;

	private:
		DArrayList<uint> mPolygonIndex;

		//Map vertex id to polygon id
		DArrayList<uint> mVer2Poly;

		//Map polygon id to edge id
		DArrayList<uint> mPoly2Edg;

		//Map edge id to triangle id
		DArray<::dyno::TopologyModule::Edg2Poly> mEdg2Poly;
	};
}

