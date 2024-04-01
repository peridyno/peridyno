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
		using Edge = TopologyModule::Edge;

		EdgeSet();
		~EdgeSet() override;

		void setEdges(const DArray<Edge>& edges);
		void setEdges(const std::vector<Edge>& edges);

		DArray<Edge>& getEdges() { return mEdgeIndex; }

		void copyFrom(EdgeSet& es);

	protected:
		/**
		 * Override updateEdges to update edges in a special way
		 */
		virtual void updateEdges() {};

		void updateTopology() override;
		VkReduce<int>& reduce() {
            return mReduce;
        };
        VkScan<int> scan() {
            return mScan;
        };

	public:
		DArray<Edge> mEdgeIndex;
	private:
        VkReduce<int> mReduce;
        VkScan<int> mScan;
	};

	using EdgeSet3f = EdgeSet<DataType3f>;
}

