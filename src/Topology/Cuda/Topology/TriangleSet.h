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
		DArray<Tri2Edg>& getTriangle2Edge() { return tri2Edg; }
		DArray<Edg2Tri>& getEdge2Triangle() { return edg2Tri; }

		/**
		 * @brief update the index from triangle id to edges ids
		 */
		void updateTriangle2Edge();

		void updateEdgeNormal(DArray<Coord>& edgeNormal);
		void updateAngleWeightedVertexNormal(DArray<Coord>& vertexNormal);
		DArray<Coord>& getEdgeNormal() { return m_edgeNormal; }
		DArray<Coord>& getVertexNormal() { return m_vertexNormal; }


		void loadObjFile(std::string filename);

		void copyFrom(TriangleSet<TDataType>& triangleSet);

		std::shared_ptr<TriangleSet<TDataType>> 
			merge(TriangleSet<TDataType>& ts);

		bool isEmpty() override;

	public:
		DEF_ARRAY_OUT(Coord, VertexNormal, DeviceType::GPU, "");

	protected:
		void updateTopology() override;

		void updateEdges() override;

		virtual void updateTriangles() {};
		virtual void updateVertexNormal();

	private:
		DArray<Triangle> mTriangleIndex;
		DArrayList<int> mVer2Tri;

		DArray<::dyno::TopologyModule::Edg2Tri> edg2Tri;
		DArray<::dyno::TopologyModule::Tri2Edg> tri2Edg;

		DArray<Coord> m_edgeNormal;
		DArray<Coord> m_vertexNormal;//Angle weighted vertex normal
	};
}

