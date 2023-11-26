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
		DArray<Triangle>& getTriangles() { return mTriangleIndex; }
		DArrayList<int>& getVertex2Triangles();

		DArray<TopologyModule::Tri2Edg>& getTriangle2Edge() { return mTri2Edg; }
		DArray<TopologyModule::Edg2Tri>& getEdge2Triangle() { return mEdg2Tri; }

		void setNormals(DArray<Coord>& normals);
		DArray<Coord>& getVertexNormals() { return mVertexNormal; }

		/**
		 * @brief update the index from triangle id to edges ids
		 */
		void updateTriangle2Edge();

		void updateEdgeNormal(DArray<Coord>& edgeNormal);
		void updateAngleWeightedVertexNormal(DArray<Coord>& vertexNormal);


		bool loadObjFile(std::string filename);

		void copyFrom(TriangleSet<TDataType>& triangleSet);

		std::shared_ptr<TriangleSet<TDataType>> 
			merge(TriangleSet<TDataType>& ts);

		bool isEmpty() override;

		//If true, normals will be updated automatically as calling update();
		void setAutoUpdateNormals(bool b) { bAutoUpdateNormal = b; }

		void rotate(const Coord angle) override;
		void rotate(const Quat<Real> q) override;

	protected:
		void updateTopology() override;

		void updateEdges() override;

		virtual void updateTriangles() {};
		virtual void updateVertexNormal();

	private:
		//A tag used to identify when the normals should be updated automatically
		bool bAutoUpdateNormal = true;

		DArray<Triangle> mTriangleIndex;
		DArrayList<int> mVer2Tri;

		DArray<::dyno::TopologyModule::Edg2Tri> mEdg2Tri;
		DArray<::dyno::TopologyModule::Tri2Edg> mTri2Edg;

		DArray<Coord> mVertexNormal;
	};
}

