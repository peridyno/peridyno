#pragma once
#include "EdgeSet.h"
#include "Catalyzer/VkSort.h"

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
			id[3] = EMPTY;
		}

		DYN_FUNC TKey(PointType v0, PointType v1, PointType v2)
		{
			id[0] = v0;
			id[1] = v1;
			id[2] = v2;
			id[3] = EMPTY;

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

		// expand to 4 bytes
		PointType id[4];
	};

	template<typename TDataType>
	class TriangleSet : public EdgeSet<TDataType>
	{
	public:
		using Coord = typename TDataType::Coord;
		using Edge = TopologyModule::Edge;
		using Triangle = TopologyModule::Triangle;
		using Tri2Edg = TopologyModule::Tri2Edg;
		using Edg2Tri = TopologyModule::Edg2Tri;

		TriangleSet();
		~TriangleSet() override;

		void setTriangles(const std::vector<Triangle>& indices);
		void setTriangles(const DArray<Triangle>& indices);

		DArray<Triangle>& getTriangles() { return mTriangleIndex; }

		//TODO: fix the hack
		DArray<uint32_t>& getVulkanIndex() { return mIndex; }

		std::shared_ptr<TriangleSet<TDataType>> merge(TriangleSet<TDataType>& ts);

		bool loadObjFile(std::string filename);

	protected:
		void updateTopology() override;
		void updateEdges() override;
		virtual void updateTriangles();

	public:
		DArray<Triangle> mTriangleIndex;
		DArray<uint32_t> mIndex;
		DArray<Edg2Tri> mEdg2Tri;
        VkSortByKey<EKey, int> mSort;
	};

	using TriangleSet3f = TriangleSet<DataType3f>;
}

