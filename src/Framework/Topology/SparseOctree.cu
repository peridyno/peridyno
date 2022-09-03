#pragma once
#include "SparseOctree.h"

#include <thrust/sort.h>

#include "Object.h"

namespace dyno {
	DYN_FUNC OcKey CalculateMortonCode(Level l, OcIndex x, OcIndex y, OcIndex z)
	{
		OcKey key = 0;

		x |= 1U << l;
		y |= 1U << l;
		z |= 1U << l;

		for (int i = 0; i < MAX_LEVEL; i++)
		{
			key |= (x & 1U << i) << 2 * i | (y & 1U << i) << (2 * i + 1) | (z & 1U << i) << (2 * i + 2);
		}
		return key;
	}


	DYN_FUNC void RecoverFromMortonCode(OcKey key, Level& l, OcIndex& x, OcIndex& y, OcIndex& z)
	{

	}

	void print(DArray<OctreeNode>& d_arr)
	{
		CArray<OctreeNode> h_arr;
		h_arr.resize(d_arr.size());

		h_arr.assign(d_arr);

		for (uint i = 0; i < h_arr.size(); i++)
		{
			Level level;
			OcIndex tnx, tny, tnz;
			h_arr[i].getCoord(level, tnx, tny, tnz);
			//std::cout << "Poster order: " << i << " " << h_arr[i].key() << " " << tnx << " " << tny << " " << tnz << std::endl;
			printf("Node %d: key: %d - %d: %d %d %d : ", i, h_arr[i].m_key, level, tnx, tny, tnz);
			printf(" Data size: %d ; start loc: %d; data loc: %d; first child: %d ", h_arr[i].getDataSize(), h_arr[i].getStartIndex(), h_arr[i].m_data_loc, h_arr[i].m_first_child_loc);

			printf("\n      child: ");
			for (int j = 0; j < 8; j++)
			{
				printf(" %d: %d ", j, h_arr[i].childs[j]);
			}

			printf("\n");
		}
		h_arr.clear();

		printf("\n\n");
	}

	DYN_FUNC OctreeNode::OctreeNode()
		: m_key(0)
		, m_level(0)
	{
		for (int i = 0; i < 8; i++)
		{
			childs[i] = EMPTY;
		}
	}

	DYN_FUNC OctreeNode::OctreeNode(OcKey key)
		: m_key(key)
		, m_level(0)
	{
		while (key != 0)
		{
			key = key >> 3U;
			m_level++;
		}

		m_level = m_level == 0 ? 0 : m_level - 1;

		for (int i = 0; i < 8; i++)
		{
			childs[i] = EMPTY;
		}
	}

	DYN_FUNC OctreeNode::OctreeNode(Level l, OcIndex x, OcIndex y, OcIndex z)
		: m_key(0)
		, m_level(l)
	{
		m_key = CalculateMortonCode(l, x, y, z);

		for (int i = 0; i < 8; i++)
		{
			childs[i] = EMPTY;
		}
	}


	DYN_FUNC void OctreeNode::getCoord(Level& l, OcIndex& x, OcIndex& y, OcIndex& z)
	{
		l = m_level;

		x = 0;
		y = 0;
		z = 0;
		for (int i = 0; i < MAX_LEVEL; i++)
		{
			x |= (m_key & (1U << DIM * i)) >> 2 * i;
			y |= (m_key & (1U << DIM * i + 1)) >> (2 * i + 1);
			z |= (m_key & (1U << DIM * i + 2)) >> (2 * i + 2);
		}

		x &= ((1U << l) - 1);
		y &= ((1U << l) - 1);
		z &= ((1U << l) - 1);
	}


	DYN_FUNC bool OctreeNode::operator==(const OctreeNode& mc2) const
	{
		return m_key == mc2.m_key;
	}

	DYN_FUNC bool OctreeNode::operator>=(const OctreeNode& mc2) const
	{
		if (mc2.isContainedStrictlyIn(*this))
			return false;

		auto k1 = m_key;
		auto k2 = mc2.m_key;

		m_level > mc2.m_level ? (k1 = k1 >> DIM * (m_level - mc2.m_level)) : (k2 = k2 >> DIM * (mc2.m_level - m_level));

		return k1 >= k2;
	}

	DYN_FUNC bool OctreeNode::operator>(const OctreeNode& mc2) const
	{
		if (isContainedStrictlyIn(mc2))
		{
			return true;
		}

		auto k1 = m_key;
		auto k2 = mc2.m_key;

		m_level > mc2.m_level ? (k1 = k1 >> DIM * (m_level - mc2.m_level)) : (k2 = k2 >> DIM * (mc2.m_level - m_level));

		return k1 > k2;
	}

	DYN_FUNC bool OctreeNode::operator<=(const OctreeNode& mc2) const
	{
		return ~(*this > mc2);
	}

	DYN_FUNC bool OctreeNode::operator<(const OctreeNode& mc2) const
	{
		return ~(*this >= mc2);
	}

	DYN_FUNC bool OctreeNode::isContainedIn(const OctreeNode& mc2) const
	{
		if (m_level < mc2.m_level)
		{
			return false;
		}

		auto k1 = m_key >> 3 * (m_level - mc2.m_level);
		auto k2 = mc2.key();

		return k1 == k2;
	}

	DYN_FUNC bool OctreeNode::isContainedStrictlyIn(const OctreeNode& mc2) const
	{
		if (m_level <= mc2.m_level)
		{
			return false;
		}

		auto k1 = m_key >> DIM * (m_level - mc2.m_level);
		auto k2 = mc2.key();

		return k1 == k2;
	}


	DYN_FUNC OctreeNode OctreeNode::leastCommonAncestor(const OctreeNode& mc2) const
	{
		OcKey k1 = m_key;
		OcKey k2 = mc2.m_key;

		m_level > mc2.m_level ? (k1 = k1 >> DIM * (m_level - mc2.m_level)) : (k2 = k2 >> DIM * (mc2.m_level - m_level));

		while (k1 != k2)
		{
			k1 = k1 >> 3U;
			k2 = k2 >> 3U;
		}
		return OctreeNode(k1);
	}

	// 	DYN_FUNC MortonCode3D& MortonCode3D::operator=(const MortonCode3D & mc2)
	// 	{
	// 		m_key = mc2.m_key;
	// 		return *this;
	// 	}
	// 
	// 	DYN_FUNC MortonCode3D MortonCode3D::operator=(const MortonCode3D & mc2) const
	// 	{
	// 		return MortonCode3D(mc2.m_key);
	// 	}

	DYN_FUNC bool OctreeNode::operator==(const OcKey k) const
	{
		return m_key == k;
	}

	DYN_FUNC bool OctreeNode::operator!=(const OcKey k) const
	{
		return m_key != k;
	}


	template<typename TDataType>
	SparseOctree<TDataType>::SparseOctree()
	{
	}

	template<typename TDataType>
	SparseOctree<TDataType>::~SparseOctree()
	{
	}

	template<typename TDataType>
	void SparseOctree<TDataType>::release()
	{
		m_all_nodes.clear();
		m_post_ordered_nodes.clear();

		node_buffer.clear();
		node_count.clear();
		nonRepeatNodes_cpy.clear();
		aux_nodes.clear();
		duplicates_count.clear();
	}

	template<typename TDataType>
	void SparseOctree<TDataType>::setSpace(Coord lo, Real h, Real L)
	{
		m_lo = lo;
		m_h = h;
		m_L = L;

		Real segments = m_L / h;

		m_level_max = ceil(log2(segments));
		//if (m_level_max < 1) m_level_max = 1;
	}

	template<typename Real, typename Coord>
	__global__ void SO_ConstructAABB(
		DArray<AABB> aabb,
		DArray<Coord> pos,
		Real radius)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= pos.size()) return;

		Coord p = pos[tId];

		aabb[tId].v0 = p - radius;
		aabb[tId].v1 = p + radius;
	}


	template<typename TDataType>
	CPU_FUNC OctreeNode SparseOctree<TDataType>::queryNode(Level l, OcIndex x, OcIndex y, OcIndex z)
	{
		OcKey key = CalculateMortonCode(l, x, y, z);

		OcKey mask = 7U << 3 * l;

		CArray<OctreeNode> h_ordered_nodes(m_post_ordered_nodes.size());
		h_ordered_nodes.assign(m_post_ordered_nodes);

		OctreeNode node = h_ordered_nodes[h_ordered_nodes.size() - 1];

		//root: level 0, i should be indexed from 1 to l
		for (int i = 1; i <= l; i++)
		{
			OcKey mask_i = mask >> DIM * i;
			int child_index = (key & mask_i) >> (DIM * (l - i));

			if (node.childs[child_index] == EMPTY)
			{
				break;
			}
			else
			{
				node = h_ordered_nodes[node.childs[child_index]];
			}
		}

		h_ordered_nodes.clear();

		if (key == node.key())
		{
			return node;
		}
		return OctreeNode();
	}

	template<typename TDataType>
	GPU_FUNC Level SparseOctree<TDataType>::requestLevelNumber(const AABB box)
	{
		return SO_ComputeLevel(box, m_h, m_level_max);
	}


	template<typename TDataType>
	GPU_FUNC int SparseOctree<TDataType>::requestIntersectionNumber(const AABB box)
	{
		Level level;

		int nx_lo, ny_lo, nz_lo;
		int nx_hi, ny_hi, nz_hi;

		int num = SO_ComputeRange(level, nx_lo, nx_hi, ny_lo, ny_hi, nz_lo, nz_hi, box, m_lo, m_h, m_level_max);

		int ret_num = 0;
		//box.intersect(box);

		if (num > 0)
		{
			for (int i = nx_lo; i <= nx_hi; i++)
				for (int j = ny_lo; j <= ny_hi; j++)
					for (int k = nz_lo; k <= nz_hi; k++)
					{
						OcKey mc = CalculateMortonCode(level, i, j, k);

						int d_num = this->requestIntersectionNumber(mc, level);

						ret_num += d_num;
					}
		}

		return ret_num;
	}

	template<typename TDataType>
	GPU_FUNC void SparseOctree<TDataType>::reqeustIntersectionIds(int* ids, const AABB box)
	{
		Level level;

		int nx_lo, ny_lo, nz_lo;
		int nx_hi, ny_hi, nz_hi;

		int num = SO_ComputeRange(level, nx_lo, nx_hi, ny_lo, ny_hi, nz_lo, nz_hi, box, m_lo, m_h, m_level_max);

		if (num > 0)
		{
			int shift = 0;
			for (int i = nx_lo; i <= nx_hi; i++)
				for (int j = ny_lo; j <= ny_hi; j++)
					for (int k = nz_lo; k <= nz_hi; k++)
					{
						OcKey mc = CalculateMortonCode(level, i, j, k);



						this->reqeustIntersectionIds(ids, shift, mc, level);
					}
		}
	}



	template<typename TDataType>
	GPU_FUNC int SparseOctree<TDataType>::requestIntersectionNumberFromLevel(const AABB box, int level)
	{
		int nx_lo, ny_lo, nz_lo;
		int nx_hi, ny_hi, nz_hi;

		int num = SO_ComputeRangeAtLevel(nx_lo, nx_hi, ny_lo, ny_hi, nz_lo, nz_hi, box, level, m_lo, m_h, m_level_max);

		int ret_num = 0;

		//printf("num = %d\n",num);

		if (num > 0)
		{
			for (int i = nx_lo; i <= nx_hi; i++)
				for (int j = ny_lo; j <= ny_hi; j++)
					for (int k = nz_lo; k <= nz_hi; k++)
					{
						OcKey mc = CalculateMortonCode(level, i, j, k);

						int d_num = this->requestIntersectionNumber(mc, level);

						ret_num += d_num;
					}
		}
		//if(ret_num > 100)
		//printf("====\n%d\n%d %d %d\n%d %d %d\n============\n",(int)pow(Real(2), int(level)),nx_lo, ny_lo, nz_lo, nx_hi, ny_hi, nz_hi);
		return ret_num;
	}

	template<typename TDataType>
	GPU_FUNC int SparseOctree<TDataType>::requestIntersectionNumberFromLevel(const AABB box, AABB* data, int level)
	{
		int nx_lo, ny_lo, nz_lo;
		int nx_hi, ny_hi, nz_hi;

		int num = SO_ComputeRangeAtLevel(nx_lo, nx_hi, ny_lo, ny_hi, nz_lo, nz_hi, box, level, m_lo, m_h, m_level_max);

		int ret_num = 0;

		//printf("num = %d\n",num);

		if (num > 0)
		{
			for (int i = nx_lo; i <= nx_hi; i++)
				for (int j = ny_lo; j <= ny_hi; j++)
					for (int k = nz_lo; k <= nz_hi; k++)
					{
						OcKey mc = CalculateMortonCode(level, i, j, k);

						int d_num = this->requestIntersectionNumber(mc, level, box, data);

						ret_num += d_num;
					}
		}
		//if(ret_num > 100)
		//printf("====\n%d\n%d %d %d\n%d %d %d\n============\n",(int)pow(Real(2), int(level)),nx_lo, ny_lo, nz_lo, nx_hi, ny_hi, nz_hi);
		return ret_num;
	}

	template<typename TDataType>
	GPU_FUNC void SparseOctree<TDataType>::reqeustIntersectionIdsFromLevel(int* ids, const AABB box, int level)
	{
		int nx_lo, ny_lo, nz_lo;
		int nx_hi, ny_hi, nz_hi;

		int num = SO_ComputeRangeAtLevel(nx_lo, nx_hi, ny_lo, ny_hi, nz_lo, nz_hi, box, level, m_lo, m_h, m_level_max);


		if (num > 0)
		{
			int shift = 0;
			for (int i = nx_lo; i <= nx_hi; i++)
				for (int j = ny_lo; j <= ny_hi; j++)
					for (int k = nz_lo; k <= nz_hi; k++)
					{
						OcKey mc = CalculateMortonCode(level, i, j, k);

						//printf("level of point: %d %d %d %d\n", level, i, j, k);
						this->reqeustIntersectionIds(ids, shift, mc, level);
					}
		}
	}

	template<typename TDataType>
	GPU_FUNC void SparseOctree<TDataType>::reqeustIntersectionIdsFromLevel(int* ids, const AABB box, AABB* data, int level)
	{
		int nx_lo, ny_lo, nz_lo;
		int nx_hi, ny_hi, nz_hi;

		int num = SO_ComputeRangeAtLevel(nx_lo, nx_hi, ny_lo, ny_hi, nz_lo, nz_hi, box, level, m_lo, m_h, m_level_max);


		if (num > 0)
		{
			int shift = 0;
			for (int i = nx_lo; i <= nx_hi; i++)
				for (int j = ny_lo; j <= ny_hi; j++)
					for (int k = nz_lo; k <= nz_hi; k++)
					{
						OcKey mc = CalculateMortonCode(level, i, j, k);

						//printf("level of point: %d %d %d %d\n", level, i, j, k);
						this->reqeustIntersectionIds(ids, shift, mc, level, box, data);
					}
		}
	}


	template<typename TDataType>
	GPU_FUNC int SparseOctree<TDataType>::requestIntersectionNumberFromBottom(const AABB box)
	{
		return requestIntersectionNumberFromLevel(box, m_level_max);
	}

	template<typename TDataType>
	GPU_FUNC void SparseOctree<TDataType>::reqeustIntersectionIdsFromBottom(int* ids, const AABB box)
	{
		reqeustIntersectionIdsFromLevel(ids, box, m_level_max);
	}

	template<typename TDataType>
	GPU_FUNC int SparseOctree<TDataType>::requestIntersectionNumberFromBottom(const AABB box, AABB* data)
	{
		return requestIntersectionNumberFromLevel(box, data, m_level_max);
	}

	template<typename TDataType>
	GPU_FUNC void SparseOctree<TDataType>::reqeustIntersectionIdsFromBottom(int* ids, const AABB box, AABB* data)
	{
		reqeustIntersectionIdsFromLevel(ids, box, data, m_level_max);
	}

	template<typename TDataType>
	GPU_FUNC int SparseOctree<TDataType>::requestIntersectionNumber(const OcKey key, const Level l)
	{
		int ret_num = 0;

		OcKey mask = 7U << DIM * l;

		OctreeNode node = m_post_ordered_nodes[m_post_ordered_nodes.size() - 1];

		ret_num += node.getDataSize();


		int lp = node.level();
		//root: level 0, i should be indexed from 1 to l
		for (int i = lp + 1; i <= l;)
		{
			OcKey mask_i = mask >> DIM * i;
			int child_index = (key & mask_i) >> (DIM * (l - i));

			if (node.childs[child_index] == EMPTY)
			{
				break;
			}
			else
			{
				node = m_post_ordered_nodes[node.childs[child_index]];

				int lc = node.level();
				if (lc - lp > 1)
				{
					auto k1 = key >> DIM * (l - lc);
					auto k2 = node.key();
					if (!(k1 == k2)) break;
				}

				ret_num += node.getDataSize();
				//the octree may be incomplete, skip over missing internal nodes.

				i += (lc - lp);
				lp = lc;
			}
		}

		return ret_num;
	}

	template<typename TDataType>
	GPU_FUNC int SparseOctree<TDataType>::requestIntersectionNumber(const OcKey key, const Level l, const AABB box, AABB* data)
	{
		int ret_num = 0;

		OcKey mask = 7U << DIM * l;

		OctreeNode node = m_post_ordered_nodes[m_post_ordered_nodes.size() - 1];

		ret_num += node.getDataSize();


		int lp = node.level();
		//root: level 0, i should be indexed from 1 to l
		for (int i = lp + 1; i <= l;)
		{
			OcKey mask_i = mask >> DIM * i;
			int child_index = (key & mask_i) >> (DIM * (l - i));

			if (node.childs[child_index] == EMPTY)
			{
				break;
			}
			else
			{
				node = m_post_ordered_nodes[node.childs[child_index]];

				int lc = node.level();
				if (lc - lp > 1)
				{
					auto k1 = key >> DIM * (l - lc);
					auto k2 = node.key();
					if (!(k1 == k2)) break;
				}
				int t_id = node.getStartIndex();
				int nd_size = node.getDataSize();
				AABB tmp_box;

				for (int t = 0; t < nd_size; t++)
				{
					if (box.intersect(data[m_all_nodes[t_id + t].getDataIndex()], tmp_box))
						ret_num++;
				}
				//the octree may be incomplete, skip over missing internal nodes.

				i += (lc - lp);
				lp = lc;
			}
		}

		return ret_num;
	}


	template<typename TDataType>
	GPU_FUNC void SparseOctree<TDataType>::reqeustIntersectionIds(int* ids, int& shift, const OcKey key, const Level l)
	{
		OcKey mask = 7U << DIM * l;

		OctreeNode node = m_post_ordered_nodes[m_post_ordered_nodes.size() - 1];

		int nd_size = node.getDataSize();

		int t_id = node.getStartIndex();
		for (int t = 0; t < nd_size; t++)
		{
			ids[shift] = m_all_nodes[t_id + t].getDataIndex();
			shift++;
		}
		//printf("level = %d, shift = %d\n", 0, shift);
		int lp = node.level();
		//root: level 0, i should be indexed from 1 to l
		for (int i = lp + 1; i <= l;)
		{

			OcKey mask_i = mask >> DIM * i;
			int child_index = (key & mask_i) >> (DIM * (l - i));
			//printf("child_index = %d\n", child_index);
			if (node.childs[child_index] == EMPTY)
			{
				break;
			}
			else
			{
				node = m_post_ordered_nodes[node.childs[child_index]];
				int nd_size = node.getDataSize();

				int lc = node.level();

				if (lc - lp > 1)
				{
					auto k1 = key >> DIM * (l - lc);
					auto k2 = node.key();
					if (!(k1 == k2)) break;
				}

				int t_id = node.getStartIndex();
				for (int t = 0; t < nd_size; t++)
				{
					ids[shift] = m_all_nodes[t_id + t].getDataIndex();
					shift++;
				}

				//the octree may be incomplete, skip over missing internal nodes.

				i += (lc - lp);
				lp = lc;

				Level level;
				OcIndex x, y, z;
				node.getCoord(level, x, y, z);
				//printf("level = %d, shift = %d, level = %d, x = %d, y = %d, z = %d\n", i, shift, level, x, y, z);
			}


		}
		//	printf("shift = %d\n", shift);
	}

	template<typename TDataType>
	GPU_FUNC void SparseOctree<TDataType>::reqeustIntersectionIds(int* ids, int& shift, const OcKey key, const Level l, const AABB box, AABB* data)
	{
		OcKey mask = 7U << DIM * l;

		OctreeNode node = m_post_ordered_nodes[m_post_ordered_nodes.size() - 1];

		int nd_size = node.getDataSize();

		int t_id = node.getStartIndex();
		for (int t = 0; t < nd_size; t++)
		{
			ids[shift] = m_all_nodes[t_id + t].getDataIndex();
			shift++;
		}
		//printf("level = %d, shift = %d\n", 0, shift);
		int lp = node.level();
		//root: level 0, i should be indexed from 1 to l
		for (int i = lp + 1; i <= l;)
		{

			OcKey mask_i = mask >> DIM * i;
			int child_index = (key & mask_i) >> (DIM * (l - i));
			//printf("child_index = %d\n", child_index);
			if (node.childs[child_index] == EMPTY)
			{
				break;
			}
			else
			{
				node = m_post_ordered_nodes[node.childs[child_index]];
				int nd_size = node.getDataSize();

				int lc = node.level();
				if (lc - lp > 1)
				{
					auto k1 = key >> DIM * (l - lc);
					auto k2 = node.key();
					if (!(k1 == k2)) break;
				}

				int t_id = node.getStartIndex();
				AABB tmp_box;
				for (int t = 0; t < nd_size; t++)
				{
					if (box.intersect(data[m_all_nodes[t_id + t].getDataIndex()], tmp_box))
					{
						ids[shift] = m_all_nodes[t_id + t].getDataIndex();
						shift++;
					}
				}

				//the octree may be incomplete, skip over missing internal nodes.

				i += (lc - lp);
				lp = lc;

				Level level;
				OcIndex x, y, z;
				node.getCoord(level, x, y, z);
				//printf("level = %d, shift = %d, level = %d, x = %d, y = %d, z = %d\n", i, shift, level, x, y, z);
			}


		}
		//	printf("shift = %d\n", shift);
	}
	template<typename TDataType>
	void SparseOctree<TDataType>::construct(
		DArray<Coord>& points,
		Real radius)
	{
		DArray<AABB> aabb;
		aabb.resize(points.size());


		cuExecute(points.size(),
			SO_ConstructAABB,
			aabb,
			points,
			radius);

		this->construct(aabb);


		aabb.clear();
	}

	template<typename Real>
	DYN_FUNC inline Level SO_ComputeLevel(
		const AABB box,
		Real h_min,
		int level_max)
	{
		Real len_max = maximum(box.length(0), maximum(box.length(1), box.length(2)));
		//len_max /= 4.0f;
		return level_max - clamp(int(ceil(log2(len_max / h_min))), 0, level_max);
	}

	template<typename Coord>
	DYN_FUNC int SO_ComputeRange(
		Level& level,
		int& nx_lo,
		int& nx_hi,
		int& ny_lo,
		int& ny_hi,
		int& nz_lo,
		int& nz_hi,
		AABB box,
		Coord origin,
		Real h_min,
		int level_max)
	{
		Coord lo_rel = box.v0 - origin;
		Coord hi_rel = box.v1 - origin;

		level = SO_ComputeLevel(box, h_min, level_max);

		int grid_size = (int)pow(Real(2), int(level));
		Real h = h_min * pow(Real(2), level_max - level);

		// 		nx_lo = clamp(int(floor(lo_rel[0] / h)), 0, grid_size - 1);
		// 		ny_lo = clamp(int(floor(lo_rel[1] / h)), 0, grid_size - 1);
		// 		nz_lo = clamp(int(floor(lo_rel[2] / h)), 0, grid_size - 1);
		// 
		// 		nx_hi = clamp(int(floor(hi_rel[0] / h)), 0, grid_size - 1);
		// 		ny_hi = clamp(int(floor(hi_rel[1] / h)), 0, grid_size - 1);
		// 		nz_hi = clamp(int(floor(hi_rel[2] / h)), 0, grid_size - 1);

		nx_lo = (int)floor(lo_rel[0] / h);
		ny_lo = (int)floor(lo_rel[1] / h);
		nz_lo = (int)floor(lo_rel[2] / h);

		nx_hi = (int)floor(hi_rel[0] / h);
		ny_hi = (int)floor(hi_rel[1] / h);
		nz_hi = (int)floor(hi_rel[2] / h);




		if (nx_hi < 0 || nx_lo >= grid_size || ny_hi < 0 || ny_lo >= grid_size || nz_hi < 0 || nz_lo >= grid_size)
		{
			return 0;
		}


		nx_lo = clamp(nx_lo, 0, grid_size - 1);
		ny_lo = clamp(ny_lo, 0, grid_size - 1);
		nz_lo = clamp(nz_lo, 0, grid_size - 1);

		nx_hi = clamp(nx_hi, 0, grid_size - 1);
		ny_hi = clamp(ny_hi, 0, grid_size - 1);
		nz_hi = clamp(nz_hi, 0, grid_size - 1);

		return (nz_hi - nz_lo + 1) * (ny_hi - ny_lo + 1) * (nx_hi - nx_lo + 1);
	}

	template<typename Coord>
	DYN_FUNC int SO_ComputeRangeAtLevel(
		int& nx_lo,
		int& nx_hi,
		int& ny_lo,
		int& ny_hi,
		int& nz_lo,
		int& nz_hi,
		AABB box,
		int level,
		Coord origin,
		Real h_min,
		int level_max)
	{
		Coord lo_rel = box.v0 - origin;
		Coord hi_rel = box.v1 - origin;

		int grid_size = (int)pow(Real(2), int(level));
		Real h = h_min * pow(Real(2), level_max - level);

		nx_lo = (int)floor(lo_rel[0] / h);
		ny_lo = (int)floor(lo_rel[1] / h);
		nz_lo = (int)floor(lo_rel[2] / h);

		nx_hi = (int)floor(hi_rel[0] / h);
		ny_hi = (int)floor(hi_rel[1] / h);
		nz_hi = (int)floor(hi_rel[2] / h);

		if (nx_hi < 0 || nx_lo >= grid_size || ny_hi < 0 || ny_lo >= grid_size || nz_hi < 0 || nz_lo >= grid_size)
		{
			return 0;
		}

		nx_lo = clamp(nx_lo, 0, grid_size - 1);
		ny_lo = clamp(ny_lo, 0, grid_size - 1);
		nz_lo = clamp(nz_lo, 0, grid_size - 1);

		nx_hi = clamp(nx_hi, 0, grid_size - 1);
		ny_hi = clamp(ny_hi, 0, grid_size - 1);
		nz_hi = clamp(nz_hi, 0, grid_size - 1);

		return (nz_hi - nz_lo + 1) * (ny_hi - ny_lo + 1) * (nx_hi - nx_lo + 1);
	}


	template<typename Coord>
	__global__ void SO_InitCounter(
		DArray<int> counter,
		DArray<AABB> aabb,
		Coord origin,
		Real h_min,
		int level_max)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= aabb.size()) return;

		Level level;

		int nx_lo;
		int ny_lo;
		int nz_lo;

		int nx_hi;
		int ny_hi;
		int nz_hi;

		counter[tId] = SO_ComputeRange(level, nx_lo, nx_hi, ny_lo, ny_hi, nz_lo, nz_hi, aabb[tId], origin, h_min, level_max);
	}

	template<typename Coord>
	__global__ void SO_CreateAllNodes(
		DArray<OctreeNode> nodes,
		DArray<AABB> aabb,
		DArray<int> index,
		Coord origin,
		Real h_min,
		int level_max)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= aabb.size()) return;

		Level level;

		int nx_lo;
		int ny_lo;
		int nz_lo;

		int nx_hi;
		int ny_hi;
		int nz_hi;

		int num = SO_ComputeRange(level, nx_lo, nx_hi, ny_lo, ny_hi, nz_lo, nz_hi, aabb[tId], origin, h_min, level_max);

		if (num > 0)
		{
			int acc_num = 0;
			for (int i = nx_lo; i <= nx_hi; i++)
				for (int j = ny_lo; j <= ny_hi; j++)
					for (int k = nz_lo; k <= nz_hi; k++)
					{
						auto mc = OctreeNode(level, i, j, k);
						mc.setDataIndex(tId);

						nodes[index[tId] + acc_num] = mc;

						//printf("%d %d %d %d %d \n", index[tId], morton[index[tId] + acc_num].childs[0], i, j, k);

						acc_num++;
					}
		}
	}

	__global__ void SO_CountNonRepeatedNodes(
		DArray<int> counter,
		DArray<OctreeNode> morton)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= counter.size()) return;

		if ((tId == 0 || morton[tId].key() != morton[tId - 1].key()))
		{
			counter[tId] = 1;
		}
		else
			counter[tId] = 0;

		//printf("%d %d \n", tId, counter[tId]);
	}

	//the size of counter is 1 more than the size of post_ordered_nodes
	__global__ void SO_RemoveDuplicativeNodes(
		DArray<OctreeNode> non_repeated_nodes,
		DArray<OctreeNode> post_ordered_nodes,
		DArray<int> counter
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= post_ordered_nodes.size()) return;

		if (tId == 0 || post_ordered_nodes[tId].key() != post_ordered_nodes[tId - 1].key())
		{
			non_repeated_nodes[counter[tId]] = post_ordered_nodes[tId];
			non_repeated_nodes[counter[tId]].setStartIndex(tId);
		}

		//printf("%d %d \n", tId, counter[tId]);
	}

	__global__ void	SO_CalculateDataSize(
		DArray<OctreeNode> non_repeated_nodes,
		int non_repeated_node_num,
		int total_node_num)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= non_repeated_node_num) return;

		if (tId < non_repeated_node_num - 1)
			non_repeated_nodes[tId].setDataSize(non_repeated_nodes[tId + 1].getStartIndex() - non_repeated_nodes[tId].getStartIndex());
		else
			non_repeated_nodes[tId].setDataSize(total_node_num - non_repeated_nodes[tId].getStartIndex());
	}

	__global__ void SO_GenerateLCA(
		DArray<OctreeNode> nodes,
		int shift
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= shift - 1) return;

		nodes[tId + shift] = nodes[tId].leastCommonAncestor(nodes[tId + 1]);
	}

	__global__ void SO_RemoveDuplicativeInternalNodes(
		DArray<OctreeNode> nodes,
		DArray<int> counter,
		DArray<OctreeNode> nodes_cpy)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= nodes.size()) return;

		if (tId > 0 && nodes[tId].key() == nodes[tId - 1].key())
		{
			counter[tId] = 0;
			nodes[tId] = OctreeNode();
		}
		else
		{
			counter[tId] = 1;
			nodes[tId] = nodes_cpy[tId];
		}
	}

	__global__ void SO_GenerateAuxNodes(
		DArray<OctreeNode> aux_nodes,
		DArray<OctreeNode> post_ordered_nodes,
		int shift)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= post_ordered_nodes.size()) return;

		aux_nodes[tId] = post_ordered_nodes[tId];
		aux_nodes[tId].m_current_loc = tId;
		if (tId < shift - 1)
		{
			auto cp = post_ordered_nodes[tId].leastCommonAncestor(post_ordered_nodes[tId + 1]);
			cp.setFirstChildIndex(tId);
			cp.m_bCopy = true;

			aux_nodes[tId + shift] = cp;
		}
	}

	__global__ void SO_GeneratePostOrderedNodes(
		DArray<OctreeNode> post_ordered_nodes,
		DArray<OctreeNode> nodes_buffer,
		int num)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= num) return;

		post_ordered_nodes[tId] = nodes_buffer[tId];
	}

	__global__ void SO_SetupParentChildRelationship(
		DArray<OctreeNode> post_ordered_nodes,
		DArray<OctreeNode> aux_nodes)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= aux_nodes.size() - 1)
			return;

		int aux_num = aux_nodes.size();

		int t = tId + 1;
		if ((aux_nodes[tId].m_bCopy != aux_nodes[t].m_bCopy))
		{
			int aux_level = aux_nodes[tId].m_level;
			int aux_key = aux_nodes[tId].key();
			int aux_cur_loc = aux_nodes[tId].m_current_loc;
			while (t < aux_num && (aux_nodes[tId].key() == aux_nodes[t].key()))
			{
				int tLoc = aux_nodes[t].getFirstChildIndex();

				OctreeNode child = post_ordered_nodes[tLoc];

				auto cKey = child.key();

				int cIndex = (cKey >> 3 * (child.m_level - aux_level - 1)) & 7U;

				post_ordered_nodes[aux_cur_loc].childs[cIndex] = tLoc;

				t++;
			}
		}
	}

	template<typename TDataType>
	void SparseOctree<TDataType>::construct(DArray<AABB>& aabb)
	{
		DArray<int> data_count;
		data_count.resize(aabb.size());


		/*************** step 1: identify nodes that containing data ****************/
		//step 1.1: calculate the cell number each point covers
		cuExecute(data_count.size(),
			SO_InitCounter,
			data_count,
			aabb,
			m_lo,
			m_h,
			m_level_max);

		//step 1.2: allocate enough space to store all cells
		int total_node_num = thrust::reduce(thrust::device, data_count.begin(), data_count.begin() + data_count.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, data_count.begin(), data_count.begin() + data_count.size(), data_count.begin());

		m_all_nodes.resize(total_node_num);


		//std::cout << MAX_LEVEL << ' ' << m_level_max << std::endl;

		//step 1.3: create all octree nodes, record the data location using node.setDataIndex().
		cuExecute(aabb.size(),
			SO_CreateAllNodes,
			m_all_nodes,
			aabb,
			data_count,
			m_lo,
			m_h,
			m_level_max);

		//print(m_all_nodes);
		//std::cout << total_node_num << std::endl;
		thrust::sort(thrust::device, m_all_nodes.begin(), m_all_nodes.begin() + m_all_nodes.size(), NodeCmp());

		//print(m_all_nodes);

		construct(m_all_nodes);

		data_count.clear();
		//thrust::sort_by_key(thrust::device, m_key.getDataPtr(), m_key.getDataPtr() + m_key.size(), m_morton.getDataPtr());
	}

	template<typename TDataType>
	void SparseOctree<TDataType>::construct(DArray<OctreeNode>& nodes)
	{
		/*************** step 2: remove duplicative nodes ****************/
		int total_node_num = nodes.size();

		duplicates_count.resize(total_node_num);

		cuExecute(duplicates_count.size(),
			SO_CountNonRepeatedNodes,
			duplicates_count,
			nodes);

		int non_duplicative_num = thrust::reduce(thrust::device, duplicates_count.begin(), duplicates_count.begin() + duplicates_count.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, duplicates_count.begin(), duplicates_count.begin() + duplicates_count.size(), duplicates_count.begin());

		node_buffer.resize(2 * non_duplicative_num - 1);

		//Remove duplicative nodes, record the first location using node.setStartIndex();
		cuExecute(nodes.size(),
			SO_RemoveDuplicativeNodes,
			node_buffer,
			nodes,
			duplicates_count);

		//print(node_buffer);

		cuExecute(non_duplicative_num,
			SO_CalculateDataSize,
			node_buffer,
			non_duplicative_num,
			total_node_num);

		//print(node_buffer);


		//thrust::sort(thrust::device, nonRepeatMorton.getDataPtr(), nonRepeatMorton.getDataPtr() + non_repeated_num, NodeCmp());

		/*************** step 3: step up parent-child relationship * ****************/

		//step 3.1: Generate internal nodes & sort
		cuExecute(non_duplicative_num - 1,
			SO_GenerateLCA,
			node_buffer,
			non_duplicative_num);

		thrust::sort(thrust::device, node_buffer.begin(), node_buffer.begin() + node_buffer.size(), NodeCmp());

		//print(node_buffer);
		nonRepeatNodes_cpy.resize(node_buffer.size());
		nonRepeatNodes_cpy.assign(node_buffer);

		//leaf + LCA
		node_count.resize(node_buffer.size());

		//step 3.2: remove duplicates
		cuExecute(node_buffer.size(),
			SO_RemoveDuplicativeInternalNodes,
			node_buffer,
			node_count,
			nonRepeatNodes_cpy);

		thrust::sort(thrust::device, node_buffer.begin(), node_buffer.begin() + node_buffer.size(), NodeCmp());
		non_duplicative_num = thrust::reduce(thrust::device, node_count.begin(), node_count.begin() + node_count.size(), (int)0, thrust::plus<int>());


		m_post_ordered_nodes.resize(non_duplicative_num);

		cuExecute(m_post_ordered_nodes.size(),
			SO_GeneratePostOrderedNodes,
			m_post_ordered_nodes,
			node_buffer,
			non_duplicative_num);

		//print(m_post_ordered_nodes);
		aux_nodes.resize(2 * m_post_ordered_nodes.size() - 1);

		cuExecute(m_post_ordered_nodes.size(),
			SO_GenerateAuxNodes,
			aux_nodes,
			m_post_ordered_nodes,
			m_post_ordered_nodes.size());

		//print(morton_b);

		thrust::sort(thrust::device, aux_nodes.begin(), aux_nodes.begin() + aux_nodes.size(), NodeCmp());

		//print(aux_nodes);

		cuExecute(aux_nodes.size(),
			SO_SetupParentChildRelationship,
			m_post_ordered_nodes,
			aux_nodes);

		//print(m_post_ordered_nodes);
	}

	template<typename TDataType>
	void SparseOctree<TDataType>::printPostOrderedTree()
	{
		print(m_post_ordered_nodes);
	}


	template<typename TDataType>
	void SparseOctree<TDataType>::printAllNodes()
	{
		print(m_all_nodes);
	}

	DEFINE_CLASS(SparseOctree);
}