#pragma once
#include "DataTypes.h"
#include "TopologyConstants.h"
#include "Primitive3D.h"
#include "Array/Array.h"

namespace dyno {

	typedef unsigned short OcIndex;
	typedef unsigned long long int OcKey;
	typedef unsigned short Level;

	typedef typename TAlignedBox3D<Real> AABB;

#define MAX_LEVEL 15
#define DIM 3

	DYN_FUNC OcKey CalculateMortonCode(Level l, OcIndex x, OcIndex y, OcIndex z);
	DYN_FUNC void RecoverFromMortonCode(OcKey key, Level& l, OcIndex& x, OcIndex& y, OcIndex& z);

	class OctreeNode
	{
	public:
		DYN_FUNC OctreeNode();
		DYN_FUNC OctreeNode(OcKey key);

		/**
		 * @brief
		 *
		 * @param l, grid level, 0 indicates the root level
		 * @param x, should be smaller than 2^l
		 * @param y, should be smaller than 2^l
		 * @param z, should be smaller than 2^l
		 * @return DYN_FUNC
		 */
		DYN_FUNC OctreeNode(Level l, OcIndex x, OcIndex y, OcIndex z);

		DYN_FUNC bool operator== (const OcKey k) const;
		DYN_FUNC bool operator!= (const OcKey k) const;

		/**
		 * @brief Octree will be traversed in the post-ordered fashion
		 *
		 * @return true if two nodes are in the post-ordered fashion
		 */
		DYN_FUNC bool operator>= (const OctreeNode&) const;

		DYN_FUNC bool operator> (const OctreeNode&) const;
		DYN_FUNC bool operator<= (const OctreeNode&) const;
		DYN_FUNC bool operator< (const OctreeNode&) const;
		DYN_FUNC bool operator== (const OctreeNode&) const;

		// 		DYN_FUNC MortonCode3D& operator= (const MortonCode3D &);
		// 		DYN_FUNC MortonCode3D operator= (const MortonCode3D &) const;

		DYN_FUNC inline bool isContainedIn(const OctreeNode&) const;
		DYN_FUNC inline bool isContainedStrictlyIn(const OctreeNode&) const;

		DYN_FUNC void getCoord(Level& l, OcIndex& x, OcIndex& y, OcIndex& z);


		DYN_FUNC OctreeNode leastCommonAncestor(const OctreeNode&) const;

		DYN_FUNC inline OcKey key() const { return m_key; }
		DYN_FUNC inline Level level() const { return m_level; }

		DYN_FUNC inline void	setDataIndex(int id) { m_data_loc = id; }
		DYN_FUNC inline int	getDataIndex() { return m_data_loc; }

		DYN_FUNC inline void	setStartIndex(int id) { m_start_loc = id; }
		DYN_FUNC inline int	getStartIndex() { return m_start_loc; }

		DYN_FUNC inline void	setFirstChildIndex(int id) { m_first_child_loc = id; }
		DYN_FUNC inline int	getFirstChildIndex() { return m_first_child_loc; }

		DYN_FUNC inline void	setDataSize(int n) { m_data_size = n; }
		DYN_FUNC inline int	getDataSize() { return m_data_size; }

		DYN_FUNC inline bool isValid() { return m_key > 0; }

		DYN_FUNC inline bool isEmpty() { return m_data_size == 0; }

	public:
		OcKey m_key;
		Level m_level;

		int m_data_loc = EMPTY;

		int m_start_loc = EMPTY;
		int m_data_size = 0;

		int m_current_loc = EMPTY;

		int m_first_child_loc = EMPTY;

		bool m_bCopy = false;

		int childs[8];
	};

	struct NodeCmp
	{
		DYN_FUNC bool operator()(const OctreeNode& A, const OctreeNode& B)
		{
			return A > B;
		}
	};

	template<typename TDataType>
	class SparseOctree
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SparseOctree();
		~SparseOctree();

		/**
		 * @brief Call release() to release allocated memory explicitly, do not call this function from the decontructor.
		 *
		 */
		void release();

		void setSpace(Coord lo, Real h, Real L);

		void construct(DArray<Coord>& points, Real radius);
		void construct(DArray<AABB>& aabb);

		void construct(DArray<OctreeNode>& nodes);

		int getLevelMax() { return m_level_max; }


		CPU_FUNC OctreeNode queryNode(Level l, OcIndex x, OcIndex y, OcIndex z);

		GPU_FUNC Level requestLevelNumber(const AABB box);

		GPU_FUNC int requestIntersectionNumber(const AABB box);
		GPU_FUNC void reqeustIntersectionIds(int* ids, const AABB box);

		GPU_FUNC int requestIntersectionNumberFromLevel(const AABB box, int level);
		GPU_FUNC int requestIntersectionNumberFromLevel(const AABB box, AABB* data, int level);
		GPU_FUNC void reqeustIntersectionIdsFromLevel(int* ids, const AABB box, int level);
		GPU_FUNC void reqeustIntersectionIdsFromLevel(int* ids, const AABB box, AABB* data, int level);

		GPU_FUNC int requestIntersectionNumberFromBottom(const AABB box);
		GPU_FUNC void reqeustIntersectionIdsFromBottom(int* ids, const AABB box);

		GPU_FUNC int requestIntersectionNumberFromBottom(const AABB box, AABB* data);
		GPU_FUNC void reqeustIntersectionIdsFromBottom(int* ids, const AABB box, AABB* data);

	public:

		void printAllNodes();
		void printPostOrderedTree();

	private:
		GPU_FUNC int requestIntersectionNumber(const OcKey key, const Level l);
		GPU_FUNC int requestIntersectionNumber(const OcKey key, const Level l, const AABB box, AABB* data);
		GPU_FUNC void reqeustIntersectionIds(int* ids, int& shift, const OcKey key, const Level l);
		GPU_FUNC void reqeustIntersectionIds(int* ids, int& shift, const OcKey key, const Level l, const AABB box, AABB* data);


	private:
		/**
		 * @brief levels are numbered from 0 to m_level_max;
		 *
		 */
		int m_level_max;

		Real m_h;
		Real m_L;

		Coord m_lo;

		DArray<OctreeNode> m_all_nodes;
		DArray<OctreeNode> m_post_ordered_nodes;

		DArray<int> duplicates_count;
		DArray<int> node_count;
		DArray<OctreeNode> aux_nodes;
		DArray<OctreeNode> node_buffer;
		DArray<OctreeNode> nonRepeatNodes_cpy;
	};
}
