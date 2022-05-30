#pragma once
#include "Algorithm/Scan.h"
#include "Algorithm/Reduction.h"

#include "SparseOctree.h"

namespace dyno
{
	template<typename TDataType>
	class SparseGridHash
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SparseGridHash();
		~SparseGridHash();

		void setSpace(Coord lo, Real h, Real L);

		void construct(DArray<Coord>& points, Real h);

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

		SparseOctree<TDataType> mSparseOctree;
	};
}
