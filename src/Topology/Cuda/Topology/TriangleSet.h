/**
 * Copyright 2017-2023 Xiaowei He
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
#include "EdgeSet.h"

namespace dyno
{
	class TKey
	{
	public:
		DYN_FUNC TKey()
		{
			id[0] = EMPTY;
			id[1] = EMPTY;
			id[2] = EMPTY;
		}

		DYN_FUNC TKey(PointType v0, PointType v1, PointType v2)
		{
			id[0] = v0;
			id[1] = v1;
			id[2] = v2;

			swap(id[0], id[1]);
			swap(id[0], id[2]);
			swap(id[1], id[2]);
		}

		DYN_FUNC inline PointType operator[] (unsigned int i) { return id[i]; }
		DYN_FUNC inline PointType operator[] (unsigned int i) const { return id[i]; }

		DYN_FUNC inline bool operator>= (const TKey& other) const {
			if (id[0] >= other.id[0]) return true;
			if (id[0] == other.id[0] && id[1] >= other.id[1]) return true;
			if (id[0] == other.id[0] && id[1] == other.id[1] && id[2] >= other.id[2]) return true;

			return false;
		}

		DYN_FUNC inline bool operator> (const TKey& other) const {
			if (id[0] > other.id[0]) return true;
			if (id[0] == other.id[0] && id[1] > other.id[1]) return true;
			if (id[0] == other.id[0] && id[1] == other.id[1] && id[2] > other.id[2]) return true;

			return false;
		}

		DYN_FUNC inline bool operator<= (const TKey& other) const {
			if (id[0] <= other.id[0]) return true;
			if (id[0] == other.id[0] && id[1] <= other.id[1]) return true;
			if (id[0] == other.id[0] && id[1] == other.id[1] && id[2] <= other.id[2]) return true;

			return false;
		}

		DYN_FUNC inline bool operator< (const TKey& other) const {
			if (id[0] < other.id[0]) return true;
			if (id[0] == other.id[0] && id[1] < other.id[1]) return true;
			if (id[0] == other.id[0] && id[1] == other.id[1] && id[2] < other.id[2]) return true;

			return false;
		}

		DYN_FUNC inline bool operator== (const TKey& other) const {
			return id[0] == other.id[0] && id[1] == other.id[1] && id[2] == other.id[2];
		}

		DYN_FUNC inline bool operator!= (const TKey& other) const {
			return id[0] != other.id[0] || id[1] != other.id[1] || id[2] != other.id[2];
		}

	private:
		DYN_FUNC inline void swap(PointType& v0, PointType& v1)
		{
			PointType vt = v0;
			v0 = v0 < v1 ? v0 : v1;
			v1 = vt < v1 ? v1 : vt;
		}

		PointType id[3];
	};

	enum VertexNormalWeightingOption
	{
		UNIFORM_WEIGHTED,
		AREA_WEIGHTED,
		ANGLE_WEIGHTED
	};

	template<typename TDataType>
	class TriangleSet : public EdgeSet<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		TriangleSet();
		~TriangleSet() override;

		void setTriangles(std::vector<Triangle>& triangles);
		void setTriangles(DArray<Triangle>& triangles);

		/**
		 * @brief return all triangle indices
		 */
		DArray<Triangle>& triangleIndices() { return mTriangleIndex; }

		const DArrayList<int>& vertex2Triangle() {return mVer2Tri; }

		const DArray<TopologyModule::Tri2Edg>& triangle2Edge() { return mTri2Edg; }
		const DArray<TopologyModule::Edg2Tri>& edge2Triangle() { return mEdg2Tri; }


		bool loadObjFile(std::string filename);

		void copyFrom(TriangleSet<TDataType>& triangleSet);

		std::shared_ptr<TriangleSet<TDataType>> 
			merge(TriangleSet<TDataType>& ts);

		bool isEmpty() override;

		void clear() override;

	public:
		void requestTriangle2Triangle(DArray<::dyno::TopologyModule::Tri2Tri>& t2t);

		void requestEdgeNormals(DArray<Coord>& normals);

		// Calculate vertex normals according to the given option op
		void requestVertexNormals(DArray<Coord>& normals, VertexNormalWeightingOption op = VertexNormalWeightingOption::ANGLE_WEIGHTED);

	protected:
		// Update mTriangleIndex, mVer2Tri
		void updateTopology() override;

		// Update {Edges, Tri2Edg, Edg2Tri}
		void updateEdges() override;
		
		virtual void updateTriangles() {}
	private:
		// Update {Ver2Tri}
		void updateVertex2Triangle();

		DArray<Triangle> mTriangleIndex;

		// A mapping from vertex ids to triangle ids,  automatically updated when update() is called
		DArrayList<int> mVer2Tri;

		// Mapping between edges and triangles, automatically updated when update() is called
		DArray<::dyno::TopologyModule::Edg2Tri> mEdg2Tri;
		DArray<::dyno::TopologyModule::Tri2Edg> mTri2Edg;
	};
}

