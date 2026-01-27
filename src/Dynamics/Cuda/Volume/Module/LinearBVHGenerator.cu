#include "LinearBVHGenerator.h"
#include "Algorithm/Scan.h"
#include "Algorithm/Reduction.h"
#include <thrust/sort.h>
#include <ctime>
#include "Timer.h"
#include "STL/Stack.h"

namespace dyno
{
	IMPLEMENT_TCLASS(LinearBVHGenerator, TDataType)

	__global__ void LBVH_InitialBinaryRadixTree(
		DArray<vBVHNode> bvhNodes,
		DArray<OcKey> finest_nodes,
		Level level,
		int b_num)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= b_num) return;

		if (i == (b_num - 1))
			bvhNodes[i + b_num - 1] = vBVHNode(level, finest_nodes[i]);
		else
		{
			bvhNodes[i + b_num - 1] = vBVHNode(level, finest_nodes[i]);
			bvhNodes[i] = vBVHNode();
		}
	}

	__global__ void LBVH_ConstructBinaryRadixTree(
		DArray<vBVHNode> bvhNodes,
		int b_num)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= b_num - 1) return;

		//Calculate the length of the longest common prefix between i and j, note i should be in the range of [b_num-1, 2*b_num-2]
		auto delta = [&](int _i, int _j) -> int {
			if (_j < 0 || _j >= b_num) return -1;
			return __clzll(bvhNodes[_i + b_num - 1].m_morton ^ bvhNodes[_j + b_num - 1].m_morton);
		};

		int d = delta(i, i + 1) - delta(i, i - 1) > 0 ? 1 : -1;
		// Compute upper bound for the length of the range
		int delta_min = delta(i, i - d);

		// Find the other end using binary search
		int len_max = 2;
		while (delta(i, i + len_max * d) > delta_min)
		{
			len_max *= 2;
		}

		int len = 0;
		for (int t = len_max / 2; t > 0; t = t / 2)
		{
			if (delta(i, i + (len + t) * d) > delta_min)
			{
				len = len + t;
			}
		}
		int j = i + len * d;

		// Find the split position using binary search
		int delta_node = delta(i, j);
		int s = 0;
		for (int t = (len + 1) / 2; t > 0; t = t == 1 ? 0 : (t + 1) / 2)
		{
			if (delta(i, i + (s + t) * d) > delta_node)
			{
				s = s + t;
			}
		}
		int gamma = i + s * d + minimum(d, (int)0);

		//Output child pointers
		int left_idx = minimum(i, j) == gamma ? gamma + b_num - 1 : gamma;
		int right_idx = maximum(i, j) == gamma + 1 ? gamma + b_num : gamma + 1;
		//compute the level and morton code of intermediate nodes
		int post_morton = 8 * sizeof(OcKey) - delta_node;

		if (post_morton % 3 > 0)
			post_morton += (3 - post_morton % 3);
		Level i_l = (bvhNodes[i + b_num - 1].m_level) - post_morton / 3;
		OcKey i_m = (bvhNodes[i + b_num - 1].m_morton) >> post_morton;

		bvhNodes[i].m_level = i_l;
		bvhNodes[i].m_morton = i_m;

		bvhNodes[i].left = left_idx;
		bvhNodes[i].right = right_idx;

		bvhNodes[left_idx].parent = i;
		bvhNodes[right_idx].parent = i;
	}

	__global__ void LBVH_CountChains(
		DArray<uint> count,
		DArray<vBVHNode> nodes)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= count.size()) return;

		if (tId == 0) return;

		int c_num = nodes[tId].m_level - nodes[nodes[tId].parent].m_level;
		if (c_num > 1)
			count[tId] = c_num - 1;
	}

	__global__ void LBVH_ComputeChains(
		DArray<vBVHNode> nodes,
		DArray<uint> count,
		int lbvh_num)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= count.size()) return;

		if (tId == 0) return;

		int c_num = nodes[tId].m_level - nodes[nodes[tId].parent].m_level;
		if (c_num > 1)
		{
			int findex = nodes[tId].parent;
			int cindex = tId;
			for (int i = 1; i < c_num; i++)
			{
				vBVHNode mc(nodes[tId].m_level - i, nodes[tId].m_morton >> (3 * i));
				mc.left = cindex;
				int pindex = lbvh_num + count[tId] + c_num - 1 - i;
				nodes[pindex] = mc;
				nodes[cindex].parent = pindex;
				cindex = pindex;
			}
			nodes[cindex].parent = findex;
			if (nodes[findex].left == tId)
				nodes[findex].left = cindex;
			else
				nodes[findex].right = cindex;
		}
	}

	__global__ void LBVH_CountCoverLeafs(
		DArray<uint> count,
		DArray<vBVHNode> nodes,
		Level l_)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= count.size()) return;

		if (nodes[tId].m_level == l_) return;

		if (tId == 0 || (nodes[tId].m_level != nodes[nodes[tId].parent].m_level))
			count[tId] = 1;
	}

	template <typename Real, typename Coord>
	__global__ void LBVH_ComputeCoverLeafs(
		DArray<AdaptiveGridNode> nodes,
		DArray<uint> count,
		DArray<vBVHNode> bvh_nodes,
		Coord origin,
		Real dx,
		Level max_level,
		int bvh_num)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= count.size()) return;

		if (bvh_nodes[tId].m_level == max_level) return;

		if (tId == 0 || (bvh_nodes[tId].m_level != bvh_nodes[bvh_nodes[tId].parent].m_level))
		{
			Level level = bvh_nodes[tId].m_level + 1;
			OcKey morton = (bvh_nodes[tId].m_morton) << 3;

			OcIndex gnx, gny, gnz;
			RecoverFromMortonCode(morton, gnx, gny, gnz);
			Real up_dx = dx * (1<<(max_level - level));
			Coord pos(origin[0] + (gnx + 0.5) * up_dx, origin[1] + (gny + 0.5) * up_dx, origin[2] + (gnz + 0.5) * up_dx);

			nodes[8 * count[tId]] = AdaptiveGridNode(level, morton, pos);
			AdaptiveGridNode mc1(level, morton + 1, pos + Coord(up_dx, 0, 0));
			nodes[8 * count[tId] + 1] = mc1;
			AdaptiveGridNode mc2(level, morton + 2, pos + Coord(0, up_dx, 0));
			nodes[8 * count[tId] + 2] = mc2;
			AdaptiveGridNode mc3(level, morton + 3, pos + Coord(up_dx, up_dx, 0));
			nodes[8 * count[tId] + 3] = mc3;
			AdaptiveGridNode mc4(level, morton + 4, pos + Coord(0, 0, up_dx));
			nodes[8 * count[tId] + 4] = mc4;
			AdaptiveGridNode mc5(level, morton + 5, pos + Coord(up_dx, 0, up_dx));
			nodes[8 * count[tId] + 5] = mc5;
			AdaptiveGridNode mc6(level, morton + 6, pos + Coord(0, up_dx, up_dx));
			nodes[8 * count[tId] + 6] = mc6;
			AdaptiveGridNode mc7(level, morton + 7, pos + Coord(up_dx, up_dx, up_dx));
			nodes[8 * count[tId] + 7] = mc7;

			if (level < max_level)
			{
				int buffer[8];
				Stack<int> stack;
				stack.reserve(buffer, 8);
				stack.push(tId);
				while (!stack.empty())
				{
					int idx = stack.top();
					stack.pop();

					if (bvh_nodes[idx].m_level < level)
					{
						stack.push(bvh_nodes[idx].left);
						if (bvh_nodes[idx].right != EMPTY)
							stack.push(bvh_nodes[idx].right);
					}
					//else if (bvh_nodes[idx].m_level == max_level)
					//{
					//	//int resid_idx = (bvh_nodes[idx].m_morton - morton);
					//	//sdf[8 * count[tId] + resid_idx] = bvh_sdf[idx - (bvh_num - 1)];
					//	//object[8 * count[tId] + resid_idx] = bvh_object[idx - (bvh_num - 1)];
					//	//fim_count[8 * count[tId] + resid_idx] = 1;
					//}
					else if (bvh_nodes[idx].m_level == level)
					{
						int resid_idx = (bvh_nodes[idx].m_morton - morton);
						nodes[8 * count[tId] + resid_idx].m_fchild = 8 * count[idx];
					}
				}
			}
		}
	}

	template<typename TDataType>
	void LinearBVHGenerator<TDataType>::compute()
	{
		GTimer timer;
		timer.start();

		auto& m_seed = this->inpMorton()->getData();
		auto m_AGrid = this->inAGridSet()->getDataPtr();
		auto& nodes = m_AGrid->adaptiveGridNode();
		Real m_dx = m_AGrid->adaptiveGridDx();
		Coord m_origin = m_AGrid->adaptiveGridOrigin();
		Level m_levelmax = m_AGrid->adaptiveGridLevelMax();
		m_AGrid->setOctreeType((int)3);
		//std::clock_t Time1 = clock();

		int LBVH_leafnum = m_seed.size();
		//binary radix tree is used to assist the construction of globally paved leafa nodes
		DArray<vBVHNode> LBVH_nodes(2 * LBVH_leafnum - 1);
		cuExecute(LBVH_leafnum,
			LBVH_InitialBinaryRadixTree,
			LBVH_nodes,
			m_seed,
			m_levelmax,
			LBVH_leafnum);
		cuExecute(LBVH_leafnum,
			LBVH_ConstructBinaryRadixTree,
			LBVH_nodes,
			LBVH_leafnum);

		DArray<uint> data_count(2 * LBVH_leafnum - 1);
		data_count.reset();
		//complement the chains
		cuExecute(2 * LBVH_leafnum - 1,
			LBVH_CountChains,
			data_count,
			LBVH_nodes);

		Reduction<uint> reduce;
		int chain_num = reduce.accumulate(data_count.begin(), data_count.size());
		Scan<uint> scan;
		scan.exclusive(data_count.begin(), data_count.size());
		chain_num += (2 * LBVH_leafnum - 1);
		printf("LBVH and Chain num:  %d %d \n", (2 * LBVH_leafnum - 1), chain_num);

		if (chain_num > (2 * LBVH_leafnum - 1))
		{
			DArray<vBVHNode> chain_nodes;
			chain_nodes.assign(LBVH_nodes);
			LBVH_nodes.resize(chain_num);
			LBVH_nodes.assign(chain_nodes, (2 * LBVH_leafnum - 1), 0, 0);
			chain_nodes.clear();
			cuExecute(2 * LBVH_leafnum - 1,
				LBVH_ComputeChains,
				LBVH_nodes,
				data_count,
				2 * LBVH_leafnum - 1);
		}

		data_count.resize(chain_num);
		data_count.reset();
		//construct leaf nodes that cover the whole space
		cuExecute(chain_num,
			LBVH_CountCoverLeafs,
			data_count,
			LBVH_nodes,
			m_levelmax);
		int cover_num = reduce.accumulate(data_count.begin(), data_count.size());
		scan.exclusive(data_count.begin(), data_count.size());

		nodes.resize(cover_num * 8);
		nodes.reset();
		cuExecute(chain_num,
			LBVH_ComputeCoverLeafs,
			nodes,
			data_count,
			LBVH_nodes,
			m_origin,
			m_dx,
			m_levelmax,
			LBVH_leafnum);

		LBVH_nodes.clear();
		data_count.clear();

		timer.stop();
		printf("LinearBVH time: %f\n", timer.getElapsedTime());

		AdaptiveGridGenerator<TDataType>::compute();
	}

	DEFINE_CLASS(LinearBVHGenerator);
}