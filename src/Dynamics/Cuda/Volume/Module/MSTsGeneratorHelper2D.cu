#include "MSTsGeneratorHelper2D.h"
#include "Algorithm/Scan.h"
#include "Algorithm/Reduction.h"
#include <thrust/sort.h>
#include <ctime>
#include "Timer.h"
#include "STL/Stack.h"

namespace dyno
{
	__constant__ int verification2D[16][2] = {
		0, 0,
		1, 0,
		0, 1,
		1, 1,
		-1, 0,
		-1, 1,
		0, -1,
		1, -1,
		2, 0,
		2, 1,
		0, 2,
		1, 2,
		-1, -1,
		2, -1,
		2, 2,
		-1, 2 };

	GPU_FUNC bool MST2D_HashAdd(
		OcKey key,
		DArray<OcKey>& nodes)
	{
		int index = (key * 100003) % (nodes.size());
		while (atomicCAS(&(nodes[index]), (OcKey)0, key))
		{
			if (nodes[index] == key)
				return false;
			else
				index = ((++index) % (nodes.size()));
		};
		return true;
	}
	GPU_FUNC bool MST2D_HashAddAndCount(
		OcKey key,
		int knum,
		DArray<OcKey>& nodes,
		DArray<int>& count)
	{
		int index = (key * 100003) % (nodes.size());
		while (atomicCAS(&(nodes[index]), (OcKey)0, key))
		{
			if (nodes[index] == key)
				return false;
			else
				index = ((++index) % (nodes.size()));
		};
		count[index] = knum;
		return true;
	}
	GPU_FUNC int MST2D_HashCountSub(
		OcKey key,
		DArray<OcKey>& nodes,
		DArray<int>& count)
	{
		int index = (key * 100003) % (nodes.size());
		while (true)
		{
			if (nodes[index] == key)
				return ((atomicSub(&(count[index]), (int)1)) - 1);
			else if (nodes[index] == (OcKey)0)
				return 1;

			index = ((++index) % (nodes.size()));
		}
	}
	GPU_FUNC bool MST2D_HashAddCount(
		OcKey key,
		DArray<OcKey>& nodes,
		DArray<int>& count)
	{
		int index = (key * 100003) % (nodes.size());
		while (atomicCAS(&(nodes[index]), (OcKey)0, key))
		{
			OcKey mKey = nodes[index];
			int nCount = count[index];

			if (mKey == key && nCount > 0)
				return false;
			else if (mKey == key && nCount < 1)
			{
				count[index] = 1;
				return true;
			}
			else
				index = ((++index) % (nodes.size()));
		};
		count[index] = 1;
		return true;
	}
	GPU_FUNC void MST2D_DeletableHashDelete(
		OcKey key,
		DArray<OcKey>& nodes,
		DArray<bool>& count)
	{
		int index = (key * 100003) % (nodes.size());
		while (true)
		{
			OcKey index_now = nodes[index];
			if (index_now == key)
			{
				count[index] = true;
				return;
			}
			else if (index_now == (OcKey)0)	return;

			index = ((++index) % (nodes.size()));
		}
		return;
	}
	GPU_FUNC bool MST2D_DeletableHashAccess(
		OcKey key,
		DArray<OcKey>& nodes,
		DArray<bool>& count)
	{
		int index = (key * 100003) % (nodes.size());
		while (true)
		{
			OcKey index_now = nodes[index];
			bool isdelete = count[index];
			if (index_now == key && isdelete == false)
				return true;
			else if (index_now == key && isdelete == true)
				return false;
			else if (index_now == (OcKey)0)
				return false;

			index = ((++index) % (nodes.size()));
		}
	}
	GPU_FUNC void MST2D_DeletableHashDeleteUndo(
		OcKey key,
		DArray<OcKey>& nodes,
		DArray<bool>& count)
	{
		int index = (key * 100003) % (nodes.size());
		while (true)
		{
			OcKey index_now = nodes[index];
			if (index_now == key)
			{
				count[index] = false;
				return;
			}
			else if (index_now == (OcKey)0)	return;

			index = ((++index) % (nodes.size()));
		}
		return;
	}

	//__global__ void CFS_ConstructNodesBufferIndexOld(
	//	DArray<OcKey> nodes_buf,
	//	DArray<OcKey> finest_nodes,
	//	Level levelmin,
	//	Level levelmax,
	//	int octreeType,
	//	int resolution)
	//{
	//	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	//	if (tId >= ((levelmax - levelmin) * finest_nodes.size())) return;

	//	Level level_offset = tId / (finest_nodes.size());
	//	int index = tId - level_offset * (finest_nodes.size());
	//	if (index > 0 && ((finest_nodes[index] >> ((level_offset + 1) * 2)) == (finest_nodes[index - 1] >> ((level_offset + 1) * 2)))) return;
	//	OcKey nmorton = finest_nodes[index] >> ((level_offset + 1) * 2);
	//	Level nlevel = levelmax - (level_offset + 1);
	//	resolution = resolution >> (level_offset + 1);

	//	//Calculate the flip object one level up
	//	auto delta = [&](int& _buffer, int _nx, int _ny, int _start_nx, int _start_ny) -> void {
	//		int _tnx, _tny;
	//		if ((_nx & 1) == 0)
	//			_tnx = (_nx >> 1) - 1;
	//		else
	//			_tnx = (_nx >> 1) + 1;
	//		if ((_ny & 1) == 0)
	//			_tny = (_ny >> 1) - 1;
	//		else
	//			_tny = (_ny >> 1) + 1;

	//		int _off_nx = _tnx - _start_nx;
	//		int _off_ny = _tny - _start_ny;
	//		int _fix_nx = (_nx >> 1) - _start_nx;
	//		int _fix_ny = (_ny >> 1) - _start_ny;

	//		_buffer = _buffer | (1 << ((_fix_ny + 1) * 3 + (_off_nx + 1)));
	//		_buffer = _buffer | (1 << ((_off_ny + 1) * 3 + (_fix_nx + 1)));
	//		if (octreeType == 0)
	//			_buffer = _buffer | (1 << ((_off_ny + 1) * 3 + (_off_nx + 1)));
	//		};

	//	if (nlevel == levelmin)
	//	{
	//		MST2D_HashAdd(((nmorton << (2 * (levelmax - levelmin) + 5)) | levelmin), nodes_buf);
	//		return;
	//	}

	//	if (octreeType == 3)
	//	{
	//		MST2D_HashAdd(((nmorton << (2 * (levelmax - nlevel) + 5)) | nlevel), nodes_buf);
	//		return;
	//	}
	//	else
	//	{
	//		MST2D_HashAdd(((nmorton << (2 * (levelmax - nlevel) + 5)) | nlevel), nodes_buf);
	//		int buffer1 = 0, buffer2 = 0;
	//		OcIndex nx_index, ny_index;
	//		RecoverFromMortonCode2D(nmorton, nx_index, ny_index);
	//		delta(buffer1, nx_index, ny_index, nx_index >> 1, ny_index >> 1);
	//		Level tlevel = nlevel - 1;
	//		while (tlevel > levelmin)
	//		{
	//			nx_index = nx_index >> 1;
	//			ny_index = ny_index >> 1;
	//			resolution = resolution >> 1;

	//			for (index = 0; index < 9; index++)
	//			{
	//				if (((buffer1 >> index) & 1) == 1)
	//				{
	//					int ny = index / 3 - 1;
	//					int nx = index - 3 * (ny + 1) - 1;
	//					if ((nx_index + nx) < 0 || (nx_index + nx) >= resolution || (ny_index + ny) < 0 || (ny_index + ny) >= resolution) continue;

	//					OcKey morton_now = CalculateMortonCode2D(nx_index + nx, ny_index + ny);
	//					if (MST2D_HashAdd(((morton_now << (2 * (levelmax - tlevel) + 5)) | tlevel), nodes_buf) == true)
	//						delta(buffer2, nx_index + nx, ny_index + ny, nx_index >> 1, ny_index >> 1);
	//				}
	//			}
	//			buffer1 = buffer2;
	//			buffer2 = 0;
	//			tlevel--;
	//		}

	//		nx_index = nx_index >> 1;
	//		ny_index = ny_index >> 1;
	//		resolution = resolution >> 1;
	//		for (index = 0; index < 9; index++)
	//		{
	//			if (((buffer1 >> index) & 1) == 1)
	//			{
	//				int ny = index / 3 - 1;
	//				int nx = index - 3 * (ny + 1) - 1;
	//				if ((nx_index + nx) < 0 || (nx_index + nx) >= resolution || (ny_index + ny) < 0 || (ny_index + ny) >= resolution) continue;

	//				OcKey morton_now = CalculateMortonCode2D(nx_index + nx, ny_index + ny);
	//				MST2D_HashAdd(((morton_now << (2 * (levelmax - tlevel) + 5)) | tlevel), nodes_buf);
	//			}
	//		}
	//	}
	//}

	__global__ void CFS_ConstructNodesBufferIndex(
		DArray<OcKey> nodes_buf,
		DArray<OcKey> finest_nodes,
		Level levelmin,
		Level levelmax,
		int octreeType,
		int resolution)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= finest_nodes.size()) return;
		if (tId > 0 && ((finest_nodes[tId] >> 2) == (finest_nodes[tId - 1] >> 2))) return;

		OcKey nmorton = finest_nodes[tId] >> 2;
		Level nlevel = levelmax - 1;
		resolution = resolution >> 1;

		//Calculate the flip object one level up
		auto delta = [&](int& _buffer, int _nx, int _ny, int _start_nx, int _start_ny) -> void {
			int _tnx, _tny;
			if ((_nx & 1) == 0)
				_tnx = (_nx >> 1) - 1;
			else
				_tnx = (_nx >> 1) + 1;
			if ((_ny & 1) == 0)
				_tny = (_ny >> 1) - 1;
			else
				_tny = (_ny >> 1) + 1;

			int _off_nx = _tnx - _start_nx;
			int _off_ny = _tny - _start_ny;
			int _fix_nx = (_nx >> 1) - _start_nx;
			int _fix_ny = (_ny >> 1) - _start_ny;

			_buffer = _buffer | (1 << ((_fix_ny + 1) * 3 + (_fix_nx + 1)));
			if (octreeType < 2)
			{
				_buffer = _buffer | (1 << ((_fix_ny + 1) * 3 + (_off_nx + 1)));
				_buffer = _buffer | (1 << ((_off_ny + 1) * 3 + (_fix_nx + 1)));
			}
			if (octreeType == 0)
				_buffer = _buffer | (1 << ((_off_ny + 1) * 3 + (_off_nx + 1)));
			};

		if (nlevel == levelmin)
		{
			MST2D_HashAdd(((nmorton << (2 * (levelmax - levelmin) + 5)) | levelmin), nodes_buf);
			return;
		}

		MST2D_HashAdd(((nmorton << (2 * (levelmax - nlevel) + 5)) | nlevel), nodes_buf);
		int buffer1 = 0, buffer2 = 0;
		OcIndex nx_index, ny_index;
		RecoverFromMortonCode2D(nmorton, nx_index, ny_index);
		delta(buffer1, nx_index, ny_index, nx_index >> 1, ny_index >> 1);
		Level tlevel = nlevel - 1;
		while (tlevel > levelmin)
		{
			nx_index = nx_index >> 1;
			ny_index = ny_index >> 1;
			resolution = resolution >> 1;

			for (int index = 0; index < 9; index++)
			{
				if (((buffer1 >> index) & 1) == 1)
				{
					int ny = index / 3 - 1;
					int nx = index - 3 * (ny + 1) - 1;
					if ((nx_index + nx) < 0 || (nx_index + nx) >= resolution || (ny_index + ny) < 0 || (ny_index + ny) >= resolution) continue;

					OcKey morton_now = CalculateMortonCode2D(nx_index + nx, ny_index + ny);
					if (MST2D_HashAdd(((morton_now << (2 * (levelmax - tlevel) + 5)) | tlevel), nodes_buf) == true)
						delta(buffer2, nx_index + nx, ny_index + ny, nx_index >> 1, ny_index >> 1);
				}
			}
			buffer1 = buffer2;
			buffer2 = 0;
			tlevel--;
		}

		nx_index = nx_index >> 1;
		ny_index = ny_index >> 1;
		resolution = resolution >> 1;
		for (int index = 0; index < 9; index++)
		{
			if (((buffer1 >> index) & 1) == 1)
			{
				int ny = index / 3 - 1;
				int nx = index - 3 * (ny + 1) - 1;
				if ((nx_index + nx) < 0 || (nx_index + nx) >= resolution || (ny_index + ny) < 0 || (ny_index + ny) >= resolution) continue;

				OcKey morton_now = CalculateMortonCode2D(nx_index + nx, ny_index + ny);
				MST2D_HashAdd(((morton_now << (2 * (levelmax - tlevel) + 5)) | tlevel), nodes_buf);
			}
		}
	}

	__global__ void CFS_CountNodesBuffer(
		DArray<int> mark,
		DArray<OcKey> nodes_buf)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes_buf.size()) return;

		if (nodes_buf[tId] != (OcKey)0)
			mark[tId] = 1;
	}

	__global__ void CFS_FetchNodesNorepeat(
		DArray<OcKey> nodes,
		DArray<int> mark,
		DArray<OcKey> nodes_buf)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes_buf.size()) return;

		if ((tId == (nodes_buf.size() - 1)) && (mark[tId] < nodes.size()))
			nodes[mark[tId]] = nodes_buf[tId];
		else if (mark[tId] != mark[tId + 1])
			nodes[mark[tId]] = nodes_buf[tId];
	}

	template <typename Real, typename Coord2D>
	__global__ void CFS_ComputeNodesAll(
		DArray<AdaptiveGridNode2D> nodes,
		DArray<OcKey> nodes_morton,
		Level levelmin,
		Level levelmax,
		Coord2D origin,
		Real dx,
		int minnum,
		int gresolution)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= (nodes_morton.size() + minnum)) return;

		if (tId < minnum)
		{
			OcIndex gnx, gny;
			RecoverFromMortonCode2D(OcKey(tId), gnx, gny);
			gresolution = gresolution >> levelmin;

			Real gdx = dx * gresolution;
			Coord2D gpos(origin[0] + (gnx + 0.5)*gdx, origin[1] + (gny + 0.5)*gdx);

			nodes[tId] = AdaptiveGridNode2D(levelmin, OcKey(tId), gpos);
		}
		else
		{
			int index = tId - minnum;

			Level nlevel = nodes_morton[index] & 31;
			OcKey nmorton = (nodes_morton[index] >> (2 * (levelmax - nlevel) + 5)) << 2;
			nlevel++;

			OcIndex gnx, gny;
			RecoverFromMortonCode2D(nmorton, gnx, gny);
			gresolution = gresolution >> nlevel;
			Real gdx = dx * gresolution;
			Coord2D gpos(origin[0] + (gnx + 0.5) * gdx, origin[1] + (gny + 0.5) * gdx);

			nodes[minnum + 4 * index] = AdaptiveGridNode2D(nlevel, nmorton, gpos);
			nodes[minnum + 4 * index + 1] = AdaptiveGridNode2D(nlevel, nmorton + 1, gpos + Coord2D(gdx, 0));
			nodes[minnum + 4 * index + 2] = AdaptiveGridNode2D(nlevel, nmorton + 2, gpos + Coord2D(0, gdx));
			nodes[minnum + 4 * index + 3] = AdaptiveGridNode2D(nlevel, nmorton + 3, gpos + Coord2D(gdx, gdx));
		}
	}

	__global__ void CFS_ComputeNodesChildRelationships(
		DArray<AdaptiveGridNode2D> nodes,
		DArray<OcKey> nodes_buf,
		Level levelmin,
		Level levelmax,
		int minnum)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= (nodes_buf.size())) return;

		auto delta = [&](int _j) -> OcKey {
			if (_j < 0 || _j >= nodes_buf.size()) return 0;
			return nodes_buf[_j];
			};

		Level nlevel = nodes_buf[tId] & 31;
		OcKey nmorton = nodes_buf[tId] >> (2 * (levelmax - nlevel) + 5);

		if (nlevel == levelmin)
		{
			nodes[nmorton].m_fchild = minnum + 4 * tId;
			return;
		}

		OcKey pmorton = ((nmorton >> 2) << (2 * (levelmax - nlevel + 1) + 5)) | (nlevel - 1);
		int len_max = 2;
		while (delta(tId + len_max) >= pmorton)
		{
			len_max *= 2;
		}

		int len = 0;
		for (int t = len_max / 2; t > 0; t = t / 2)
		{
			if (delta(tId + (len + t)) >= pmorton)
			{
				len = len + t;
			}
		}
		int find = tId + len;

		int find_off = nmorton & 3;
		nodes[minnum + 4 * find + find_off].m_fchild = minnum + 4 * tId;
	}

	template<typename TDataType>
	void MSTsGeneratorHelper2D<TDataType>::ConstructionFromScratch2D(
		std::shared_ptr<AdaptiveGridSet2D<TDataType>> AGridSet,
		DArray<OcKey>& mSeed,
		Level mLevelnum,
		int mType)
	{
		auto& nodes = AGridSet->adaptiveGridNode2D();
		Real m_dx = AGridSet->adaptiveGridDx2D();
		Coord2D m_origin = AGridSet->adaptiveGridOrigin2D();
		Level m_levelmax = AGridSet->adaptiveGridLevelMax2D();
		AGridSet->setLevelNum(mLevelnum);
		if (mLevelnum > m_levelmax)
		{
			printf("MSTsGenerator2D %d %d, levelnum is big than levemax\n", mLevelnum, m_levelmax);
			return;
		}
		AGridSet->setQuadType(mType);
		int max_resolution = (1 << m_levelmax);

		int buf_num = 3 * (mSeed.size());
		buf_num += (buf_num % 2 == 0) ? 1 : 0;
		DArray<OcKey> buffer_key(buf_num);
		buffer_key.reset();

		cuExecute(mSeed.size(),
			CFS_ConstructNodesBufferIndex,
			buffer_key,
			mSeed,
			m_levelmax - mLevelnum + 1,
			m_levelmax,
			mType,
			max_resolution);

		DArray<int> buffer_mark(buf_num);
		buffer_mark.reset();
		cuExecute(buf_num,
			CFS_CountNodesBuffer,
			buffer_mark,
			buffer_key);

		Reduction<int> reduce;
		int fnode_num = reduce.accumulate(buffer_mark.begin(), buffer_mark.size());

		Scan<int> scan;
		scan.exclusive(buffer_mark.begin(), buffer_mark.size());

		DArray<OcKey> nodes_morton(fnode_num);
		cuExecute(buf_num,
			CFS_FetchNodesNorepeat,
			nodes_morton,
			buffer_mark,
			buffer_key);

		thrust::sort(thrust::device, nodes_morton.begin(), nodes_morton.begin() + nodes_morton.size(), thrust::greater<OcKey>());

		int min_resolution = (1 << (m_levelmax - mLevelnum + 1));
		int min_nodes_num = min_resolution * min_resolution;
		nodes.resize(min_nodes_num + 4 * fnode_num);
		cuExecute(min_nodes_num + fnode_num,
			CFS_ComputeNodesAll,
			nodes,
			nodes_morton,
			m_levelmax - mLevelnum + 1,
			m_levelmax,
			m_origin,
			m_dx,
			min_nodes_num,
			max_resolution);

		cuExecute(fnode_num,
			CFS_ComputeNodesChildRelationships,
			nodes,
			nodes_morton,
			m_levelmax - mLevelnum + 1,
			m_levelmax,
			min_nodes_num);

		buffer_key.clear();
		buffer_mark.clear();
		nodes_morton.clear();
	}

    template<typename TDataType>
	__global__ void DU_PutAndCountInternalNode2D(
		DArray<OcKey> nodes_buf,
		DArray<int> buf_count,
		DArray<AdaptiveGridNode2D> nodes,
		AdaptiveGridSet2D<TDataType> gridSet,
		Level levelmax,
		int octreeType,
		int resolution)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes.size()) return;
		if (nodes[tId].isLeaf()) return;

		OcKey nmorton = nodes[tId].m_morton;
		Level nlevel = nodes[tId].m_level;
		OcIndex nx_index, ny_index;
		RecoverFromMortonCode2D(nmorton, nx_index, ny_index);
		if (nlevel == (levelmax - 1))
		{
			MST2D_HashAddAndCount(((nmorton << (2 * (levelmax - nlevel) + 5)) | nlevel), 1, nodes_buf, buf_count);
			return;
		}

		int ncount = 0;
		int cindex = nodes[tId].m_fchild;
		for (int i = 0; i < 4; i++)
		{
			if (!nodes[cindex + i].isLeaf())
				ncount++;
		}
		if (octreeType == 3)
		{
			MST2D_HashAddAndCount(((nmorton << (2 * (levelmax - nlevel) + 5)) | nlevel), ncount, nodes_buf, buf_count);
			return;
		}

		resolution = resolution >> (levelmax - nlevel - 1);
		int vindex = 12;
		if (octreeType == 0) vindex = 16;
		for (int i = 4; i < vindex; i++)
		{
			if ((2 * nx_index + verification2D[i][0]) < 0 || (2 * nx_index + verification2D[i][0]) >= resolution || (2 * ny_index + verification2D[i][1]) < 0 || (2 * ny_index + verification2D[i][1]) >= resolution) continue;
			OcKey morton_temp = CalculateMortonCode2D(2 * nx_index + verification2D[i][0], 2 * ny_index + verification2D[i][1]);
			if (!(gridSet.accessRandom2D(cindex, morton_temp, nlevel + 1)))
				ncount++;
		}

		MST2D_HashAddAndCount(((nmorton << (2 * (levelmax - nlevel) + 5)) | nlevel), ncount, nodes_buf, buf_count);
		return;
	}
	
	__global__ void DU_ConstructNodesBufferDecrease2D(
		DArray<OcKey> nodes_buf,
		DArray<int> buf_count,
		DArray<OcKey> finest_nodes,
		Level levelmin,
		Level levelmax,
		int octreeType,
		int resolution)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= finest_nodes.size()) return;
		if (tId > 0 && ((finest_nodes[tId] >> 2) == (finest_nodes[tId - 1] >> 2))) return;

		OcKey nmorton = finest_nodes[tId] >> 2;
		Level nlevel = levelmax - 1;
		resolution = resolution >> 1;

		//Calculate the flip object one level up
		auto delta = [&](int& _buffer, int _nx, int _ny, int _start_nx, int _start_ny,Level le) -> void {
			int _tnx, _tny;
			if ((_nx & 1) == 0)
				_tnx = (_nx >> 1) - 1;
			else
				_tnx = (_nx >> 1) + 1;
			if ((_ny & 1) == 0)
				_tny = (_ny >> 1) - 1;
			else
				_tny = (_ny >> 1) + 1;

			int _off_nx = _tnx - _start_nx;
			int _off_ny = _tny - _start_ny;
			int _fix_nx = (_nx >> 1) - _start_nx;
			int _fix_ny = (_ny >> 1) - _start_ny;

			if (MST2D_HashCountSub((((CalculateMortonCode2D(_start_nx + _fix_nx, _start_ny + _fix_ny)) << (2 * (levelmax - le) + 5)) | le), nodes_buf, buf_count) == 0)
				_buffer = _buffer | (1 << ((_fix_ny + 1) * 3 + (_fix_nx + 1)));
			if (octreeType < 2)
			{
				if (MST2D_HashCountSub((((CalculateMortonCode2D(_start_nx + _off_nx, _start_ny + _fix_ny)) << (2 * (levelmax - le) + 5)) | le), nodes_buf, buf_count) == 0)
					_buffer = _buffer | (1 << ((_fix_ny + 1) * 3 + (_off_nx + 1)));
				if (MST2D_HashCountSub((((CalculateMortonCode2D(_start_nx + _fix_nx, _start_ny + _off_ny)) << (2 * (levelmax - le) + 5)) | le), nodes_buf, buf_count) == 0)
					_buffer = _buffer | (1 << ((_off_ny + 1) * 3 + (_fix_nx + 1)));
			}
			if (octreeType == 0)
				if (MST2D_HashCountSub((((CalculateMortonCode2D(_start_nx + _off_nx, _start_ny + _off_ny)) << (2 * (levelmax - le) + 5)) | le), nodes_buf, buf_count) == 0)
					_buffer = _buffer | (1 << ((_off_ny + 1) * 3 + (_off_nx + 1)));
			};

		if (nlevel == levelmin)
		{
			MST2D_HashCountSub(((nmorton << (2 * (levelmax - levelmin) + 5)) | levelmin), nodes_buf, buf_count);
			return;
		}

		int buffer1 = 0, buffer2 = 0;
		OcIndex nx_index, ny_index;
		RecoverFromMortonCode2D(nmorton, nx_index, ny_index);
		if (MST2D_HashCountSub(((nmorton << (2 * (levelmax - nlevel) + 5)) | nlevel), nodes_buf, buf_count) == 0)
			delta(buffer1, nx_index, ny_index, nx_index >> 1, ny_index >> 1, nlevel - 1);
		Level tlevel = nlevel - 1;
		while (tlevel > levelmin)
		{
			nx_index = nx_index >> 1;
			ny_index = ny_index >> 1;
			resolution = resolution >> 1;
			for (int index = 0; index < 9; index++)
			{
				if (((buffer1 >> index) & 1) == 1)
				{
					int ny = index / 3 - 1;
					int nx = index - 3 * (ny + 1) - 1;
					if ((nx_index + nx) < 0 || (nx_index + nx) >= resolution || (ny_index + ny) < 0 || (ny_index + ny) >= resolution) continue;

					delta(buffer2, nx_index + nx, ny_index + ny, nx_index >> 1, ny_index >> 1, tlevel - 1);
				}
			}
			buffer1 = buffer2;
			buffer2 = 0;
			tlevel--;
		}
	}

	//__global__ void DU_ConstructNodesBufferIncrease2DOld(
	//	DArray<OcKey> nodes_buf,
	//	DArray<int> nodes_count,
	//	DArray<OcKey> finest_nodes,
	//	Level levelmin,
	//	Level levelmax,
	//	int octreeType,
	//	int resolution)
	//{
	//	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	//	if (tId >= ((levelmax - levelmin) * finest_nodes.size())) return;

	//	Level level_offset = tId / (finest_nodes.size());
	//	int index = tId - level_offset * (finest_nodes.size());
	//	if (index > 0 && ((finest_nodes[index] >> ((level_offset + 1) * 2)) == (finest_nodes[index - 1] >> ((level_offset + 1) * 2)))) return;
	//	OcKey nmorton = finest_nodes[index] >> ((level_offset + 1) * 2);
	//	Level nlevel = levelmax - (level_offset + 1);
	//	resolution = resolution >> (level_offset + 1);

	//	//Calculate the flip object one level up
	//	auto delta = [&](int& _buffer, int _nx, int _ny, int _start_nx, int _start_ny) -> void {
	//		int _tnx, _tny;
	//		if ((_nx & 1) == 0)
	//			_tnx = (_nx >> 1) - 1;
	//		else
	//			_tnx = (_nx >> 1) + 1;
	//		if ((_ny & 1) == 0)
	//			_tny = (_ny >> 1) - 1;
	//		else
	//			_tny = (_ny >> 1) + 1;

	//		int _off_nx = _tnx - _start_nx;
	//		int _off_ny = _tny - _start_ny;
	//		int _fix_nx = (_nx >> 1) - _start_nx;
	//		int _fix_ny = (_ny >> 1) - _start_ny;

	//		_buffer = _buffer | (1 << ((_fix_ny + 1) * 3 + (_off_nx + 1)));
	//		_buffer = _buffer | (1 << ((_off_ny + 1) * 3 + (_fix_nx + 1)));
	//		if (octreeType == 0)
	//			_buffer = _buffer | (1 << ((_off_ny + 1) * 3 + (_off_nx + 1)));
	//		};

	//	if (nlevel == levelmin)
	//	{
	//		MST2D_HashAddCount(((nmorton << (2 * (levelmax - levelmin) + 5)) | levelmin), nodes_buf, nodes_count);
	//		return;
	//	}

	//	if (octreeType == 3)
	//	{
	//		MST2D_HashAddCount(((nmorton << (2 * (levelmax - nlevel) + 5)) | nlevel), nodes_buf, nodes_count);
	//		return;
	//	}
	//	else
	//	{
	//		MST2D_HashAddCount(((nmorton << (2 * (levelmax - nlevel) + 5)) | nlevel), nodes_buf, nodes_count);

	//		int buffer1 = 0, buffer2 = 0;
	//		OcIndex nx_index, ny_index;
	//		RecoverFromMortonCode2D(nmorton, nx_index, ny_index);
	//		delta(buffer1, nx_index, ny_index, nx_index >> 1, ny_index >> 1);
	//		Level tlevel = nlevel - 1;
	//		while (tlevel > levelmin)
	//		{
	//			nx_index = nx_index >> 1;
	//			ny_index = ny_index >> 1;
	//			resolution = resolution >> 1;

	//			for (index = 0; index < 9; index++)
	//			{
	//				if (((buffer1 >> index) & 1) == 1)
	//				{
	//					int ny = index / 3 - 1;
	//					int nx = index - 3 * (ny + 1) - 1;
	//					if ((nx_index + nx) < 0 || (nx_index + nx) >= resolution || (ny_index + ny) < 0 || (ny_index + ny) >= resolution) continue;

	//					OcKey morton_now = CalculateMortonCode2D(nx_index + nx, ny_index + ny);
	//					if (MST2D_HashAddCount(((morton_now << (2 * (levelmax - tlevel) + 5)) | tlevel), nodes_buf, nodes_count) == true)
	//						delta(buffer2, nx_index + nx, ny_index + ny, nx_index >> 1, ny_index >> 1);
	//				}
	//			}
	//			buffer1 = buffer2;
	//			buffer2 = 0;
	//			tlevel--;
	//		}

	//		nx_index = nx_index >> 1;
	//		ny_index = ny_index >> 1;
	//		resolution = resolution >> 1;
	//		for (index = 0; index < 9; index++)
	//		{
	//			if (((buffer1 >> index) & 1) == 1)
	//			{
	//				int ny = index / 3 - 1;
	//				int nx = index - 3 * (ny + 1) - 1;
	//				if ((nx_index + nx) < 0 || (nx_index + nx) >= resolution || (ny_index + ny) < 0 || (ny_index + ny) >= resolution) continue;

	//				OcKey morton_now = CalculateMortonCode2D(nx_index + nx, ny_index + ny);
	//				MST2D_HashAddCount(((morton_now << (2 * (levelmax - tlevel) + 5)) | tlevel), nodes_buf, nodes_count);
	//			}
	//		}
	//	}
	//}

	__global__ void DU_ConstructNodesBufferIncrease2D(
		DArray<OcKey> nodes_buf,
		DArray<int> nodes_count,
		DArray<OcKey> finest_nodes,
		Level levelmin,
		Level levelmax,
		int octreeType,
		int resolution)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= finest_nodes.size()) return;
		if (tId > 0 && ((finest_nodes[tId] >> 2) == (finest_nodes[tId - 1] >> 2))) return;

		OcKey nmorton = finest_nodes[tId] >> 2;
		Level nlevel = levelmax - 1;
		resolution = resolution >> 1;

		//Calculate the flip object one level up
		auto delta = [&](int& _buffer, int _nx, int _ny, int _start_nx, int _start_ny) -> void {
			int _tnx, _tny;
			if ((_nx & 1) == 0)
				_tnx = (_nx >> 1) - 1;
			else
				_tnx = (_nx >> 1) + 1;
			if ((_ny & 1) == 0)
				_tny = (_ny >> 1) - 1;
			else
				_tny = (_ny >> 1) + 1;

			int _off_nx = _tnx - _start_nx;
			int _off_ny = _tny - _start_ny;
			int _fix_nx = (_nx >> 1) - _start_nx;
			int _fix_ny = (_ny >> 1) - _start_ny;

			_buffer = _buffer | (1 << ((_fix_ny + 1) * 3 + (_fix_nx + 1)));
			if (octreeType < 2)
			{
				_buffer = _buffer | (1 << ((_fix_ny + 1) * 3 + (_off_nx + 1)));
				_buffer = _buffer | (1 << ((_off_ny + 1) * 3 + (_fix_nx + 1)));
			}
			if (octreeType == 0)
				_buffer = _buffer | (1 << ((_off_ny + 1) * 3 + (_off_nx + 1)));
			};

		if (nlevel == levelmin)
		{
			MST2D_HashAddCount(((nmorton << (2 * (levelmax - levelmin) + 5)) | levelmin), nodes_buf, nodes_count);
			return;
		}

		MST2D_HashAddCount(((nmorton << (2 * (levelmax - nlevel) + 5)) | nlevel), nodes_buf, nodes_count);

		int buffer1 = 0, buffer2 = 0;
		OcIndex nx_index, ny_index;
		RecoverFromMortonCode2D(nmorton, nx_index, ny_index);
		delta(buffer1, nx_index, ny_index, nx_index >> 1, ny_index >> 1);
		Level tlevel = nlevel - 1;
		while (tlevel > levelmin)
		{
			nx_index = nx_index >> 1;
			ny_index = ny_index >> 1;
			resolution = resolution >> 1;

			for (int index = 0; index < 9; index++)
			{
				if (((buffer1 >> index) & 1) == 1)
				{
					int ny = index / 3 - 1;
					int nx = index - 3 * (ny + 1) - 1;
					if ((nx_index + nx) < 0 || (nx_index + nx) >= resolution || (ny_index + ny) < 0 || (ny_index + ny) >= resolution) continue;

					OcKey morton_now = CalculateMortonCode2D(nx_index + nx, ny_index + ny);
					if (MST2D_HashAddCount(((morton_now << (2 * (levelmax - tlevel) + 5)) | tlevel), nodes_buf, nodes_count) == true)
						delta(buffer2, nx_index + nx, ny_index + ny, nx_index >> 1, ny_index >> 1);
				}
			}
			buffer1 = buffer2;
			buffer2 = 0;
			tlevel--;
		}

		nx_index = nx_index >> 1;
		ny_index = ny_index >> 1;
		resolution = resolution >> 1;
		for (int index = 0; index < 9; index++)
		{
			if (((buffer1 >> index) & 1) == 1)
			{
				int ny = index / 3 - 1;
				int nx = index - 3 * (ny + 1) - 1;
				if ((nx_index + nx) < 0 || (nx_index + nx) >= resolution || (ny_index + ny) < 0 || (ny_index + ny) >= resolution) continue;

				OcKey morton_now = CalculateMortonCode2D(nx_index + nx, ny_index + ny);
				MST2D_HashAddCount(((morton_now << (2 * (levelmax - tlevel) + 5)) | tlevel), nodes_buf, nodes_count);
			}
		}
	}

	__global__ void DU_CountNodesBuffer2D(
		DArray<int> nodes_count)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes_count.size()) return;

		if (nodes_count[tId] > 0)
			nodes_count[tId] = 1;
	}

	template<typename TDataType>
	void MSTsGeneratorHelper2D<TDataType>::DynamicUpdate2D(
		std::shared_ptr<AdaptiveGridSet2D<TDataType>> AGridSet,
		DArray<OcKey>& increaseSeed,
		DArray<OcKey>& decreaseSeed,
		Level mLevelnum,
		int mType)
	{
		auto& nodes = AGridSet->adaptiveGridNode2D();
		Real m_dx = AGridSet->adaptiveGridDx2D();
		Coord2D m_origin = AGridSet->adaptiveGridOrigin2D();
		Level m_levelmax = AGridSet->adaptiveGridLevelMax2D();
		AGridSet->setLevelNum(mLevelnum);
		assert(mLevelnum <= m_levelmax);
		AGridSet->setQuadType(mType);
		int max_resolution = (1 << m_levelmax);
		int num_buffer = 3 * (nodes.size() - (AGridSet->adaptiveGridLeafNum2D()));
		num_buffer += (num_buffer % 2 == 0) ? 1 : 0;

		DArray<OcKey> buffer_key(num_buffer);
		buffer_key.reset();
		DArray<int> buffer_count(num_buffer);
		buffer_count.reset();

		cuExecute(nodes.size(),
			DU_PutAndCountInternalNode2D,
			buffer_key,
			buffer_count,
			nodes,
			*AGridSet,
			m_levelmax,
			mType,
			max_resolution);

		cuExecute(decreaseSeed.size(),
			DU_ConstructNodesBufferDecrease2D,
			buffer_key,
			buffer_count,
			decreaseSeed,
			m_levelmax - mLevelnum + 1,
			m_levelmax,
			mType,
			max_resolution);

		cuExecute(increaseSeed.size(),
			DU_ConstructNodesBufferIncrease2D,
			buffer_key,
			buffer_count,
			increaseSeed,
			m_levelmax - mLevelnum + 1,
			m_levelmax,
			mType,
			max_resolution);

		cuExecute(num_buffer,
			DU_CountNodesBuffer2D,
			buffer_count);

		Reduction<int> reduce;
		int fnode_num = reduce.accumulate(buffer_count.begin(), buffer_count.size());

		Scan<int> scan;
		scan.exclusive(buffer_count.begin(), buffer_count.size());

		DArray<OcKey> nodes_morton(fnode_num);
		cuExecute(num_buffer,
			CFS_FetchNodesNorepeat,
			nodes_morton,
			buffer_count,
			buffer_key);

		thrust::sort(thrust::device, nodes_morton.begin(), nodes_morton.begin() + nodes_morton.size(), thrust::greater<OcKey>());

		int min_resolution = (1 << (m_levelmax - mLevelnum + 1));
		int min_nodes_num = min_resolution * min_resolution;
		nodes.resize(min_nodes_num + 4 * fnode_num);
		cuExecute(min_nodes_num + fnode_num,
			CFS_ComputeNodesAll,
			nodes,
			nodes_morton,
			m_levelmax - mLevelnum + 1,
			m_levelmax,
			m_origin,
			m_dx,
			min_nodes_num,
			max_resolution);

		cuExecute(fnode_num,
			CFS_ComputeNodesChildRelationships,
			nodes,
			nodes_morton,
			m_levelmax - mLevelnum + 1,
			m_levelmax,
			min_nodes_num);

		buffer_key.clear();
		buffer_count.clear();
		nodes_morton.clear();
	}

	DEFINE_CLASS(MSTsGeneratorHelper2D);

}