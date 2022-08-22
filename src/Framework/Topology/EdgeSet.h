#pragma once
#include "PointSet.h"

namespace dyno
{
	class EKey
	{
	public:
		DYN_FUNC EKey()
		{
			id[0] = EMPTY;
			id[1] = EMPTY;
		}

		DYN_FUNC EKey(PointType v0, PointType v1)
		{
			id[0] = v0;
			id[1] = v1;

			swap(id[0], id[1]);
		}

		DYN_FUNC inline PointType operator[] (unsigned int i) { return id[i]; }
		DYN_FUNC inline PointType operator[] (unsigned int i) const { return id[i]; }

		DYN_FUNC inline bool operator>= (const EKey& other) const {
			if (id[0] >= other.id[0]) return true;
			if (id[0] == other.id[0] && id[1] >= other.id[1]) return true;

			return false;
		}

		DYN_FUNC inline bool operator> (const EKey& other) const {
			if (id[0] > other.id[0]) return true;
			if (id[0] == other.id[0] && id[1] > other.id[1]) return true;

			return false;
		}

		DYN_FUNC inline bool operator<= (const EKey& other) const {
			if (id[0] <= other.id[0]) return true;
			if (id[0] == other.id[0] && id[1] <= other.id[1]) return true;

			return false;
		}

		DYN_FUNC inline bool operator< (const EKey& other) const {
			if (id[0] < other.id[0]) return true;
			if (id[0] == other.id[0] && id[1] < other.id[1]) return true;

			return false;
		}

		DYN_FUNC inline bool operator== (const EKey& other) const {
			return id[0] == other.id[0] && id[1] == other.id[1];
		}

		DYN_FUNC inline bool operator!= (const EKey& other) const {
			return id[0] != other.id[0] || id[1] != other.id[1];
		}

		DYN_FUNC inline bool isValid() const {
			return id[0] != EMPTY && id[1] != EMPTY;
		}

	private:
		DYN_FUNC inline void swap(PointType& v0, PointType& v1)
		{
			PointType vt = v0;
			v0 = v0 < v1 ? v0 : v1;
			v1 = vt < v1 ? v1 : vt;
		}

		PointType id[2];
	};

	template<typename TDataType>
	class EdgeSet : public PointSet<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Edge Edge;

		EdgeSet();
		~EdgeSet() override;

		void updatePointNeighbors() override;

		void loadSmeshFile(std::string filename);

		DArray<Edge>& getEdges() {return m_edges;}

		void setEdges(std::vector<Edge>& edges);
		void setEdges(DArray<Edge>& edges);

		void copyFrom(EdgeSet<TDataType>& edgeSet);

		DArrayList<int>& getVer2Edge();

	protected:
		DArray<Edge> m_edges;
		DArrayList<int> m_ver2Edge;
	};
}

