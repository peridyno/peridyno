#include "BottomUpGenerator.h"
#include "Algorithm/Scan.h"
#include "Algorithm/Reduction.h"
#include "Timer.h"

namespace dyno
{
	IMPLEMENT_TCLASS(BottomUpGenerator, TDataType)

	__constant__ int offset6[6][3] = {
		-1,0,0,
		1, 0, 0,
		0, -1, 0,
		0, 1, 0,
		0, 0, -1,
		0, 0, 1
	};
	__constant__ int offset18[18][3] = {
		0, 0, 1,
		0, 1, 0,
		1, 0, 0,
		0, 0, -1,
		0, -1, 0,
		-1, 0, 0,
		0, 1, 1,
		0, 1, -1,
		0, -1, 1,
		0, -1, -1,
		1, 0, 1,
		1, 0, -1,
		-1, 0, 1,
		-1, 0, -1,
		1, 1, 0,
		1, -1, 0,
		-1, 1, 0,
		-1, -1, 0 
	};
	__constant__ int offset26[26][3] = {
		0, 0, 1,
		0, 1, 0,
		1, 0, 0,
		0, 0, -1,
		0, -1, 0,
		-1, 0, 0,
		0, 1, 1,
		0, 1, -1,
		0, -1, 1,
		0, -1, -1,
		1, 0, 1,
		1, 0, -1,
		-1, 0, 1,
		-1, 0, -1,
		1, 1, 0,
		1, -1, 0,
		-1, 1, 0,
		-1, -1, 0,
		1, 1, 1,
		1, 1, -1,
		1, -1, 1,
		-1, 1, 1,
		1, -1, -1,
		-1, 1, -1,
		-1, -1, 1,
		-1, -1, -1
	};

	__global__ void BUG_CountAugmentNum(
		DArray<uint> count,
		DArray<OcKey> nodes)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes.size()) return;

		if (tId == (nodes.size() - 1))
			count[tId] = 8;
		else
		{
			OcKey mi = nodes[tId] >> 3;
			OcKey mi1 = nodes[tId + 1] >> 3;

			if (mi != mi1)
				count[tId] = 8;
		}
	}

	template <typename Real, typename Coord>
	__global__ void BUG_ComputeAugmentNumFirst(
		DArray<AdaptiveGridNode> nodes,
		DArray<uint> count,
		DArray<OcKey> mnodes,
		Coord origin,
		Real dx,
		Level level)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= mnodes.size()) return;

		if (tId == (mnodes.size() - 1) || (count[tId] != count[tId + 1]))
		{
			OcKey mi = (mnodes[tId] >> 3) << 3;
			OcIndex gnx, gny, gnz;
			RecoverFromMortonCode(mi, gnx, gny, gnz);
			Coord pos(origin[0] + (gnx + 0.5) * dx, origin[1] + (gny + 0.5) * dx, origin[2] + (gnz + 0.5) * dx);

			nodes[count[tId]] = AdaptiveGridNode(level, mi, pos);
			nodes[count[tId] + 1] = AdaptiveGridNode(level, mi + 1, pos + Coord(dx, 0, 0));
			nodes[count[tId] + 2] = AdaptiveGridNode(level, mi + 2, pos + Coord(0, dx, 0));
			nodes[count[tId] + 3] = AdaptiveGridNode(level, mi + 3, pos + Coord(dx, dx, 0));
			nodes[count[tId] + 4] = AdaptiveGridNode(level, mi + 4, pos + Coord(0, 0, dx));
			nodes[count[tId] + 5] = AdaptiveGridNode(level, mi + 5, pos + Coord(dx, 0, dx));
			nodes[count[tId] + 6] = AdaptiveGridNode(level, mi + 6, pos + Coord(0, dx, dx));
			nodes[count[tId] + 7] = AdaptiveGridNode(level, mi + 7, pos + Coord(dx, dx, dx));
		}
	}

	__global__ void BUG_ComputeUpMorton(
		DArray<OcKey> up_morton,
		DArray<AdaptiveGridNode> nodes)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes.size()) return;

		if (tId % 8 == 0)
			up_morton[tId / 8] = (nodes[tId].m_morton) >> 3;
	}

	template <typename Real, typename Coord>
	__global__ void BUG_ComputeAugmentNum(
		DArray<AdaptiveGridNode> nodes,
		DArray<uint> count,
		DArray<OcKey> mnodes,
		Coord origin,
		Real dx,
		Level level,
		Level max_level)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= mnodes.size()) return;

		if (tId == (mnodes.size() - 1) || (count[tId] != count[tId + 1]))
		{
			OcKey mi = (mnodes[tId] >> 3) << 3;
			OcIndex gnx, gny, gnz;
			RecoverFromMortonCode(mi, gnx, gny, gnz);
			Real up_dx = dx * (1 << (max_level - level));
			Coord pos(origin[0] + (gnx + 0.5) * up_dx, origin[1] + (gny + 0.5) * up_dx, origin[2] + (gnz + 0.5) * up_dx);

			nodes[count[tId]] = AdaptiveGridNode(level, mi, pos);
			nodes[count[tId] + 1] = AdaptiveGridNode(level, mi + 1, pos + Coord(up_dx, 0, 0));
			nodes[count[tId] + 2] = AdaptiveGridNode(level, mi + 2, pos + Coord(0, up_dx, 0));
			nodes[count[tId] + 3] = AdaptiveGridNode(level, mi + 3, pos + Coord(up_dx, up_dx, 0));
			nodes[count[tId] + 4] = AdaptiveGridNode(level, mi + 4, pos + Coord(0, 0, up_dx));
			nodes[count[tId] + 5] = AdaptiveGridNode(level, mi + 5, pos + Coord(up_dx, 0, up_dx));
			nodes[count[tId] + 6] = AdaptiveGridNode(level, mi + 6, pos + Coord(0, up_dx, up_dx));
			nodes[count[tId] + 7] = AdaptiveGridNode(level, mi + 7, pos + Coord(up_dx, up_dx, up_dx));
		}
	}

	__global__ void BUG_ComputeAugmentChild(
		DArray<AdaptiveGridNode> nodes,
		DArray<uint> count,
		DArray<OcKey> mnodes)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= mnodes.size()) return;

		int axis = (mnodes[tId]) & 7U;
		nodes[count[tId] + axis].m_fchild = tId * 8;
	}

	__global__ void BUG_CollectAugment(
		DArray<AdaptiveGridNode> nodes,
		DArray<AdaptiveGridNode> nodes_temp,
		DArray<AdaptiveGridNode> nodes_new,
		int nodes_num)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes.size()) return;

		if (tId < nodes_num)
			nodes[tId] = nodes_new[tId];
		else
			nodes[tId] = nodes_temp[tId - nodes_num];

		if (!nodes[tId].isLeaf())
			nodes[tId].m_fchild += nodes_num;
	}

	template <typename Real, typename Coord>
	__global__ void BUG_ComputeTopNodes(
		DArray<AdaptiveGridNode> nodes,
		Coord origin,
		Real dx,
		Level levelmin,
		Level levelmax)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes.size()) return;

		OcIndex gnx, gny, gnz;
		RecoverFromMortonCode(OcKey(tId), gnx, gny, gnz);
		Real gdx = dx * (1 << (levelmax - levelmin));
		Coord gpos(origin[0] + (gnx + 0.5)*gdx, origin[1] + (gny + 0.5)*gdx, origin[2] + (gnz + 0.5)*gdx);

		nodes[tId] = AdaptiveGridNode(levelmin, OcKey(tId), gpos);
	}

	__global__ void BUG_ComputeTopFather(
		DArray<AdaptiveGridNode> nodes_new,
		DArray<AdaptiveGridNode> nodes,
		int nodes_num)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes_num) return;

		if ((nodes[tId].m_morton & 7) == 0)
		{
			int findex = nodes[tId].m_morton >> 3;
			nodes_new[findex].m_fchild = tId;
		}
	}

	template<typename TDataType>
	void BottomUpGenerator<TDataType>::compute()
	{
		GTimer timer;
		timer.start();

		auto& m_seed = this->inpMorton()->getData();
		auto m_AGrid = this->inAGridSet()->getDataPtr();
		auto& nodes = m_AGrid->adaptiveGridNode();
		Real m_dx = m_AGrid->adaptiveGridDx();
		Coord m_origin = m_AGrid->adaptiveGridOrigin();
		Level m_levelmax = m_AGrid->adaptiveGridLevelMax();
		Level m_levelnum = this->varLevelNum()->getData();
		m_AGrid->setLevelNum(m_levelnum);
		assert(m_levelnum <= m_levelmax);

		Reduction<uint> reduce;
		Scan<uint> scan;

		int leafnum = m_seed.size();
		DArray<uint> data_count(leafnum);
		data_count.reset();
		cuExecute(leafnum,
			BUG_CountAugmentNum,
			data_count,
			m_seed);
		int node_num = reduce.accumulate(data_count.begin(), data_count.size());
		scan.exclusive(data_count.begin(), data_count.size());

		DArray<AdaptiveGridNode> nodes_down(node_num);
		cuExecute(leafnum,
			BUG_ComputeAugmentNumFirst,
			nodes_down,
			data_count,
			m_seed,
			m_origin,
			m_dx,
			m_levelmax);
		nodes.assign(nodes_down);

		Level lnum = m_levelmax - 1;
		while (lnum > (m_levelmax - m_levelnum + 1))
		{
			leafnum = node_num / 8;
			DArray<OcKey> up_morton(leafnum);
			cuExecute(node_num,
				BUG_ComputeUpMorton,
				up_morton,
				nodes_down);

			data_count.resize(leafnum);
			data_count.reset();
			cuExecute(leafnum,
				BUG_CountAugmentNum,
				data_count,
				up_morton);
			node_num = reduce.accumulate(data_count.begin(), data_count.size());
			scan.exclusive(data_count.begin(), data_count.size());

			nodes_down.resize(node_num);
			cuExecute(leafnum,
				BUG_ComputeAugmentNum,
				nodes_down,
				data_count,
				up_morton,
				m_origin,
				m_dx,
				lnum,
				m_levelmax);
			cuExecute(leafnum,
				BUG_ComputeAugmentChild,
				nodes_down,
				data_count,
				up_morton);

			DArray<AdaptiveGridNode> nodes_temp;
			nodes_temp.assign(nodes);
			nodes.resize(nodes_temp.size() + node_num);
			cuExecute(nodes.size(),
				BUG_CollectAugment,
				nodes,
				nodes_temp,
				nodes_down,
				node_num);

			up_morton.clear();
			nodes_temp.clear();
			lnum--;
		}

		int min_resolution = (1 << (m_levelmax - m_levelnum + 1));
		int min_nodes_num = min_resolution * min_resolution * min_resolution;
		nodes_down.resize(min_nodes_num);
		cuExecute(min_nodes_num,
			BUG_ComputeTopNodes,
			nodes_down,
			m_origin,
			m_dx,
			m_levelmax - m_levelnum + 1,
			m_levelmax);

		cuExecute(node_num,
			BUG_ComputeTopFather,
			nodes_down,
			nodes,
			node_num);

		DArray<AdaptiveGridNode> nodes_temp;
		nodes_temp.assign(nodes);
		nodes.resize(nodes_temp.size() + min_nodes_num);
		cuExecute(nodes.size(),
			BUG_CollectAugment,
			nodes,
			nodes_temp,
			nodes_down,
			min_nodes_num);

		nodes_temp.clear();
		data_count.clear();
		nodes_down.clear();

		ConstraintLeafsForest();

		timer.stop();
		printf("BottomUpGenerator time: %f\n", timer.getElapsedTime());

		AdaptiveGridGenerator<TDataType>::compute();
	}


	GPU_FUNC bool BUG_GetUpGridsForest(
		int& index,
		DArray<AdaptiveGridNode>& nodes,
		OcKey morton,
		Level levelmin,
		Level level)//the range of level is [levelmin,max_level]
	{
		//compute the l-th level`s morton, the range of l is [levelmin,level]
		auto alpha = [&](Level l) -> int {
			OcKey mo = morton >> (3 * (level - l));
			return (mo & 7);
		};

		int node_index = morton >> (3 * (level - levelmin));
		if (nodes[node_index].isLeaf())
		{
			index = node_index;
			return true;
		}

		node_index = nodes[node_index].m_fchild;
		for (Level i = (levelmin + 1); i <= level; i++)
		{
			node_index = node_index + alpha(i);
			if (nodes[node_index].isLeaf())
			{
				index = node_index;
				return true;
			}
			else
				node_index = nodes[node_index].m_fchild;
		}

		return false;
	}
	__global__ void BUG_CheckLeafsForest(
		DArray<uint> count,
		DArray<AdaptiveGridNode> nodes,
		Level levelmin,
		int node_num,
		int m_octreeType)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= node_num) return;

		auto alpha = [&](OcIndex i, OcIndex j, OcIndex k, Level gl) -> bool {
			int gresolution = (1 << gl);
			if (i < 0 || i >= gresolution)
				return false;
			if (j < 0 || j >= gresolution)
				return false;
			if (k < 0 || k >= gresolution)
				return false;

			return true;
		};

		if (nodes[tId].isLeaf())
		{
			Level gl = nodes[tId].m_level;
			OcIndex gnx, gny, gnz;
			RecoverFromMortonCode((OcKey)nodes[tId].m_morton, gnx, gny, gnz);

			if (m_octreeType == 2)
			{
				for (int c = 0; c < 6; c++)
				{
					if (alpha(gnx + offset6[c][0], gny + offset6[c][1], gnz + offset6[c][2], gl))
					{
						int nindex = 0;
						OcKey nmorton = CalculateMortonCode(gnx + offset6[c][0], gny + offset6[c][1], gnz + offset6[c][2]);
						if (BUG_GetUpGridsForest(nindex, nodes, nmorton, levelmin, gl))
						{
							if (gl - (nodes[nindex].m_level) > 1)
								count[nindex] = 1;
						}
					}
				}
			}
			else if (m_octreeType == 1)
			{
				for (int c = 0; c < 18; c++)
				{
					if (alpha(gnx + offset18[c][0], gny + offset18[c][1], gnz + offset18[c][2], gl))
					{
						int nindex = 0;
						OcKey nmorton = CalculateMortonCode(gnx + offset18[c][0], gny + offset18[c][1], gnz + offset18[c][2]);
						if (BUG_GetUpGridsForest(nindex, nodes, nmorton, levelmin, gl))
						{
							if (gl - (nodes[nindex].m_level) > 1)
								count[nindex] = 1;
						}
					}
				}
			}
			else if (m_octreeType == 0)
			{
				for (int c = 0; c < 26; c++)
				{
					if (alpha(gnx + offset26[c][0], gny + offset26[c][1], gnz + offset26[c][2], gl))
					{
						int nindex = 0;
						OcKey nmorton = CalculateMortonCode(gnx + offset26[c][0], gny + offset26[c][1], gnz + offset26[c][2]);
						if (BUG_GetUpGridsForest(nindex, nodes, nmorton, levelmin, gl))
						{
							if (gl - (nodes[nindex].m_level) > 1)
								count[nindex] = 1;
						}
					}
				}
			}
			else if (m_octreeType == 4)
			{
				bool refine = false;
				for (int c = 0; c < 26; c++)
				{
					if (alpha(gnx + offset26[c][0], gny + offset26[c][1], gnz + offset26[c][2], gl))
					{
						int nindex = 0;
						OcKey nmorton = CalculateMortonCode(gnx + offset26[c][0], gny + offset26[c][1], gnz + offset26[c][2]);
						if (BUG_GetUpGridsForest(nindex, nodes, nmorton, levelmin, gl))
						{
							if (gl - (nodes[nindex].m_level) > 1)
								count[nindex] = 1;
						}
						else
							refine = true;
					}
				}
				if (refine == false) return;
				for (int c = 0; c < 26; c++)
				{
					if (alpha(gnx + offset26[c][0], gny + offset26[c][1], gnz + offset26[c][2], gl))
					{
						int nindex = 0;
						OcKey nmorton = CalculateMortonCode(gnx + offset26[c][0], gny + offset26[c][1], gnz + offset26[c][2]);
						if (BUG_GetUpGridsForest(nindex, nodes, nmorton, levelmin, gl))
						{
							if (gl - (nodes[nindex].m_level) == 1)
								count[nindex] = 1;
						}
					}
				}
			}
		}
	}

	template <typename Real, typename Coord>
	__global__ void BUG_ComputeSubdivideLeafs(
		DArray<AdaptiveGridNode> nodes,
		DArray<uint> count,
		Coord origin,
		Real dx,
		Level max_level,
		int node_num,
		int check_num)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= node_num) return;

		if ((tId == (node_num - 1) && count[tId] < check_num) || (tId < (node_num - 1) && count[tId] != count[tId + 1]))
		{
			Level level = nodes[tId].m_level + 1;
			OcKey morton = (nodes[tId].m_morton) << 3;

			OcIndex gnx, gny, gnz;
			RecoverFromMortonCode(morton, gnx, gny, gnz);
			Real up_dx = dx * (1 << (max_level - level));
			Coord pos(origin[0] + (gnx + 0.5) * up_dx, origin[1] + (gny + 0.5) * up_dx, origin[2] + (gnz + 0.5) * up_dx);


			nodes[node_num + 8 * count[tId]] = AdaptiveGridNode(level, morton, pos);
			nodes[node_num + 8 * count[tId] + 1] = AdaptiveGridNode(level, morton + 1, pos + Coord(up_dx, 0, 0));
			nodes[node_num + 8 * count[tId] + 2] = AdaptiveGridNode(level, morton + 2, pos + Coord(0, up_dx, 0));
			nodes[node_num + 8 * count[tId] + 3] = AdaptiveGridNode(level, morton + 3, pos + Coord(up_dx, up_dx, 0));
			nodes[node_num + 8 * count[tId] + 4] = AdaptiveGridNode(level, morton + 4, pos + Coord(0, 0, up_dx));
			nodes[node_num + 8 * count[tId] + 5] = AdaptiveGridNode(level, morton + 5, pos + Coord(up_dx, 0, up_dx));
			nodes[node_num + 8 * count[tId] + 6] = AdaptiveGridNode(level, morton + 6, pos + Coord(0, up_dx, up_dx));
			nodes[node_num + 8 * count[tId] + 7] = AdaptiveGridNode(level, morton + 7, pos + Coord(up_dx, up_dx, up_dx));

			nodes[tId].m_fchild = node_num + 8 * count[tId];
		}
	}

	template<typename TDataType>
	void BottomUpGenerator<TDataType>::ConstraintLeafsForest()
	{//Grids that do not satisfy the 1:2 constraint are subdivided
		auto m_AGrid = this->inAGridSet()->getDataPtr();
		auto& nodes = m_AGrid->adaptiveGridNode();
		Real m_dx = m_AGrid->adaptiveGridDx();
		Coord m_origin = m_AGrid->adaptiveGridOrigin();
		Level m_levelmax = m_AGrid->adaptiveGridLevelMax();
		Level m_levelnum = m_AGrid->adaptiveGridLevelNum();
		int m_type = this->varOctreeType()->currentKey();
		m_AGrid->setOctreeType(m_type);

		int node_num = nodes.size();
		int check_num = 0;

		DArray<AdaptiveGridNode> nodes_temp;
		DArray<uint> data_count;
		Reduction<uint> reduce;
		Scan<uint> scan;
		do {
			data_count.resize(node_num);
			data_count.reset();
			cuExecute(node_num,
				BUG_CheckLeafsForest,
				data_count,
				nodes,
				m_levelmax - m_levelnum + 1,
				node_num,
				m_type);
			check_num = reduce.accumulate(data_count.begin(), data_count.size());
			scan.exclusive(data_count.begin(), data_count.size());

			if (check_num == 0) break;

			nodes_temp.resize(node_num + (check_num * 8));
			nodes_temp.assign(nodes, node_num, 0, 0);
			cuExecute(node_num,
				BUG_ComputeSubdivideLeafs,
				nodes_temp,
				data_count,
				m_origin,
				m_dx,
				m_levelmax,
				node_num,
				check_num);
			node_num += check_num * 8;
			nodes.resize(node_num);
			nodes.assign(nodes_temp);
		} while (check_num > 0);

		data_count.clear();
		nodes_temp.clear();
	}

	DEFINE_CLASS(BottomUpGenerator);
}