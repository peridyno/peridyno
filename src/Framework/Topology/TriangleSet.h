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
		~TriangleSet();

		DArray<Triangle>& getTriangles() { return m_triangles; }
		void setTriangles(std::vector<Triangle>& triangles);

		DArrayList<int>& getVertex2Triangles();

		void updateEdges();

		void updateVertexNormal();

		void loadObjFile(std::string filename);

		void copyFrom(TriangleSet<TDataType>& triangleSet);

		DEF_ARRAY_OUT(Coord, VertexNormal, DeviceType::GPU, "");

	protected:
		void updateTopology() override;

		DArray<Triangle> m_triangles;
		DArrayList<int> m_ver2Tri;

		DArray<Edg2Tri> edg2Tri;
	};
}

