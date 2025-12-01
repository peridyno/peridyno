#include "MSTsGeneratorHelper.h"
#include "Algorithm/Scan.h"
#include "Algorithm/Reduction.h"
#include <thrust/sort.h>
#include <ctime>
#include "Timer.h"
#include "STL/Stack.h"

namespace dyno
{
	__constant__ int verification[64][3] = {
	0, 0, 0,  
	1, 0, 0,
	0, 1, 0,
	1, 1, 0,
	0, 0, 1,
	1, 0, 1,
	0, 1, 1,
	1, 1, 1,
	-1, 0, 0,
	-1, 1, 0,
	-1, 0, 1,
	-1, 1, 1,
	2, 0, 0,
	2, 1, 0,
	2, 0, 1,
	2, 1, 1,
	0, -1, 0,
	1, -1, 0,
	0, -1, 1,
	1, -1, 1,
	0, 2, 0,
	0, 2, 1,
	1, 2, 0,
	1, 2, 1,
	0, 0, -1,
	0, 1, -1,
	1, 0, -1,
	1, 1, -1,
	0, 0, 2,
	0, 1, 2,
	1, 0, 2,
	1, 1, 2,
	-1, -1, 0,
	-1, -1, 1,
	-1, 2, 0,
	-1, 2, 1,
	2, -1, 0,
	2, -1, 1,
	2, 2, 0,
	2, 2, 1,
	-1, 0, -1,
	-1, 1, -1,
	2, 0, -1,
	2, 1, -1,
	-1, 0, 2,
	-1, 1, 2,
	2, 0, 2,
	2, 1, 2,
	0, -1, -1,
	1, -1, -1,
	0, 2, -1,
	1, 2, -1,
	0, -1, 2,
	1, -1, 2,
	0, 2, 2,
	1, 2, 2,
	-1, -1, -1,
	2, -1, -1,
	-1, 2, -1,
	2, 2, -1,
	-1, -1, 2,
	2, -1, 2,
	-1, 2, 2,
	2, 2, 2};

	GPU_FUNC bool MSTG_HashAdd(
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
	GPU_FUNC bool MSTG_HashAddAndCount(
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
	GPU_FUNC bool MSTG_HashAddCount(
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
	GPU_FUNC void MSTG_DeletableHashDelete(
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
	GPU_FUNC int MSTG_HashCountSub(
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
	GPU_FUNC bool MSTG_HashAccess(
		OcKey key,
		DArray<OcKey>& nodes)
	{
		int index = (key * 100003) % (nodes.size());
		while (true)
		{
			if (nodes[index] == key)
				return true;
			else if (nodes[index] == (OcKey)0)
				return false;

			index = ((++index) % (nodes.size()));
		}
	}
	GPU_FUNC bool MSTG_DeletableHashAccess(
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
	GPU_FUNC bool MSTG_DeletableHashDeleteUndo(
		OcKey key,
		DArray<OcKey>& nodes,
		DArray<bool>& count)
	{
		int index = (key * 100003) % (nodes.size());
		while (true)
		{
			if (nodes[index] == key && count[index] == true)
			{
				count[index] = false;
				return true;
			}
			else if (nodes[index] == key && count[index] == false)
				return false;
			else if (nodes[index] == 0)
			{
				printf("!!!!!!!!!!!!DeletableHashDeleteUndo  %lld %lld %d \n", key, nodes[index], count[index]);
				return false;
			}

			index = (++index) % (nodes.size());
		}
	}


	__global__ void MSTG_ConstructNodesBufferIndex(
		DArray<OcKey> nodes_buf,
		DArray<OcKey> finest_nodes,
		Level levelmin,
		Level levelmax,
		int octreeType,
		int resolution)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= ((levelmax - levelmin) * finest_nodes.size())) return;

		Level level_offset = tId / (finest_nodes.size());
		int index = tId - level_offset * (finest_nodes.size());
		if (index > 0 && ((finest_nodes[index] >> ((level_offset + 1) * 3)) == (finest_nodes[index - 1] >> ((level_offset + 1) * 3)))) return;
		OcKey nmorton = finest_nodes[index] >> ((level_offset + 1) * 3);
		Level nlevel = levelmax - (level_offset + 1);
		resolution = resolution >> (level_offset + 1);

		//Calculate the flip object one level up
		auto delta = [&](int& _buffer, int _nx, int _ny, int _nz, int _start_nx, int _start_ny, int _start_nz, int _mark) -> void {
			int _tnx, _tny, _tnz;
			if ((_mark & 1) == 0)
				_tnx = (_nx >> 1) - 1;
			else
				_tnx = (_nx >> 1) + 1;
			if ((_mark & 2) == 0)
				_tny = (_ny >> 1) - 1;
			else
				_tny = (_ny >> 1) + 1;
			if ((_mark & 4) == 0)
				_tnz = (_nz >> 1) - 1;
			else
				_tnz = (_nz >> 1) + 1;

			int _off_nx = _tnx - _start_nx;
			int _off_ny = _tny - _start_ny;
			int _off_nz = _tnz - _start_nz;
			int _fix_nx = (_nx >> 1) - _start_nx;
			int _fix_ny = (_ny >> 1) - _start_ny;
			int _fix_nz = (_nz >> 1) - _start_nz;

			_buffer = _buffer | (1 << ((_fix_nz + 1) * 9 + (_fix_ny + 1) * 3 + (_off_nx + 1)));
			_buffer = _buffer | (1 << ((_fix_nz + 1) * 9 + (_off_ny + 1) * 3 + (_fix_nx + 1)));
			_buffer = _buffer | (1 << ((_off_nz + 1) * 9 + (_fix_ny + 1) * 3 + (_fix_nx + 1)));
			if (octreeType < 2)
			{
				_buffer = _buffer | (1 << ((_fix_nz + 1) * 9 + (_off_ny + 1) * 3 + (_off_nx + 1)));
				_buffer = _buffer | (1 << ((_off_nz + 1) * 9 + (_off_ny + 1) * 3 + (_fix_nx + 1)));
				_buffer = _buffer | (1 << ((_off_nz + 1) * 9 + (_fix_ny + 1) * 3 + (_off_nx + 1)));
			}
			if (octreeType == 0)
				_buffer = _buffer | (1 << ((_off_nz + 1) * 9 + (_off_ny + 1) * 3 + (_off_nx + 1)));
			};

		if (nlevel == levelmin)
		{
			MSTG_HashAdd(((nmorton << (3 * (levelmax - levelmin) + 4)) | levelmin), nodes_buf);
			return;
		}

		//octree: non-graded
		if (octreeType == 3)
			MSTG_HashAdd(((nmorton << (3 * (levelmax - nlevel) + 4)) | nlevel), nodes_buf);
		else
		{//octree: graded 
			MSTG_HashAdd(((nmorton << (3 * (levelmax - nlevel) + 4)) | nlevel), nodes_buf);

			int buffer1 = 0, buffer2 = 0;
			OcIndex nx_index, ny_index, nz_index;
			morton3D_64_Decode_magicbits(nmorton, nx_index, ny_index, nz_index);
			delta(buffer1, nx_index, ny_index, nz_index, nx_index >> 1, ny_index >> 1, nz_index >> 1, nmorton & 7);
			Level tlevel = nlevel - 1;
			while (tlevel > levelmin)
			{
				nx_index = nx_index >> 1;
				ny_index = ny_index >> 1;
				nz_index = nz_index >> 1;
				resolution = resolution >> 1;

				for (index = 0; index < 27; index++)
				{
					if (((buffer1 >> index) & 1) == 1)
					{
						int nz = index / 9 - 1;
						int ny = (index - 9 * (nz + 1)) / 3 - 1;;
						int nx = index - 9 * (nz + 1) - 3 * (ny + 1) - 1;
						if ((nx_index + nx) < 0 || (nx_index + nx) >= resolution || (ny_index + ny) < 0 || (ny_index + ny) >= resolution || (nz_index + nz) < 0 || (nz_index + nz) >= resolution) continue;

						OcKey morton_now = morton3D_64_Encode_magicbits(nx_index + nx, ny_index + ny, nz_index + nz);

						if (MSTG_HashAdd(((morton_now << (3 * (levelmax - tlevel) + 4)) | tlevel), nodes_buf) == true)
							delta(buffer2, nx_index + nx, ny_index + ny, nz_index + nz, nx_index >> 1, ny_index >> 1, nz_index >> 1, morton_now & 7);
					}
				}
				buffer1 = buffer2;
				buffer2 = 0;
				tlevel--;
			}

			nx_index = nx_index >> 1;
			ny_index = ny_index >> 1;
			nz_index = nz_index >> 1;
			resolution = resolution >> 1;
			for (index = 0; index < 27; index++)
			{
				if (((buffer1 >> index) & 1) == 1)
				{
					int nz = index / 9 - 1;
					int ny = (index - 9 * (nz + 1)) / 3 - 1;;
					int nx = index - 9 * (nz + 1) - 3 * (ny + 1) - 1;
					if ((nx_index + nx) < 0 || (nx_index + nx) >= resolution || (ny_index + ny) < 0 || (ny_index + ny) >= resolution || (nz_index + nz) < 0 || (nz_index + nz) >= resolution) continue;

					OcKey morton_now = morton3D_64_Encode_magicbits(nx_index + nx, ny_index + ny, nz_index + nz);
					MSTG_HashAdd(((morton_now << (3 * (levelmax - tlevel) + 4)) | tlevel), nodes_buf);
				}
			}
		}
	}

	__global__ void MSTG_ConstructNodesBufferIndexSingle(
		DArray<OcKey> nodes_buf,
		DArray<OcKey> finest_nodes,
		Level levelmin,
		Level levelmax,
		int octreeType,
		int resolution)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= finest_nodes.size()) return;
		if (tId > 0 && ((finest_nodes[tId] >> 3) == (finest_nodes[tId - 1] >> 3))) return;

		OcKey nmorton = finest_nodes[tId] >> 3;
		Level nlevel = levelmax - 1;
		resolution = resolution >> 1;

		//Calculate the flip object one level up
		auto delta = [&](int& _buffer, int _nx, int _ny, int _nz, int _start_nx, int _start_ny, int _start_nz, int _mark) -> void {
			int _tnx, _tny, _tnz;
			if ((_mark & 1) == 0)
				_tnx = (_nx >> 1) - 1;
			else
				_tnx = (_nx >> 1) + 1;
			if ((_mark & 2) == 0)
				_tny = (_ny >> 1) - 1;
			else
				_tny = (_ny >> 1) + 1;
			if ((_mark & 4) == 0)
				_tnz = (_nz >> 1) - 1;
			else
				_tnz = (_nz >> 1) + 1;

			int _off_nx = _tnx - _start_nx;
			int _off_ny = _tny - _start_ny;
			int _off_nz = _tnz - _start_nz;
			int _fix_nx = (_nx >> 1) - _start_nx;
			int _fix_ny = (_ny >> 1) - _start_ny;
			int _fix_nz = (_nz >> 1) - _start_nz;

			_buffer = _buffer | (1 << ((_fix_nz + 1) * 9 + (_fix_ny + 1) * 3 + (_fix_nx + 1)));
			if (octreeType < 3)
			{
				_buffer = _buffer | (1 << ((_fix_nz + 1) * 9 + (_fix_ny + 1) * 3 + (_off_nx + 1)));
				_buffer = _buffer | (1 << ((_fix_nz + 1) * 9 + (_off_ny + 1) * 3 + (_fix_nx + 1)));
				_buffer = _buffer | (1 << ((_off_nz + 1) * 9 + (_fix_ny + 1) * 3 + (_fix_nx + 1)));
			}
			if (octreeType < 2)
			{
				_buffer = _buffer | (1 << ((_fix_nz + 1) * 9 + (_off_ny + 1) * 3 + (_off_nx + 1)));
				_buffer = _buffer | (1 << ((_off_nz + 1) * 9 + (_off_ny + 1) * 3 + (_fix_nx + 1)));
				_buffer = _buffer | (1 << ((_off_nz + 1) * 9 + (_fix_ny + 1) * 3 + (_off_nx + 1)));
			}
			if (octreeType == 0)
				_buffer = _buffer | (1 << ((_off_nz + 1) * 9 + (_off_ny + 1) * 3 + (_off_nx + 1)));
			};

		if (nlevel == levelmin)
		{
			MSTG_HashAdd(((nmorton << (3 * (levelmax - levelmin) + 4)) | levelmin), nodes_buf);
			return;
		}

		MSTG_HashAdd(((nmorton << (3 * (levelmax - nlevel) + 4)) | nlevel), nodes_buf);

		int buffer1 = 0, buffer2 = 0;
		OcIndex nx_index, ny_index, nz_index;
		morton3D_64_Decode_magicbits(nmorton, nx_index, ny_index, nz_index);
		delta(buffer1, nx_index, ny_index, nz_index, nx_index >> 1, ny_index >> 1, nz_index >> 1, nmorton & 7);
		Level tlevel = nlevel - 1;
		while (tlevel > levelmin)
		{
			nx_index = nx_index >> 1;
			ny_index = ny_index >> 1;
			nz_index = nz_index >> 1;
			resolution = resolution >> 1;

			for (int index = 0; index < 27; index++)
			{
				if (((buffer1 >> index) & 1) == 1)
				{
					int nz = index / 9 - 1;
					int ny = (index - 9 * (nz + 1)) / 3 - 1;;
					int nx = index - 9 * (nz + 1) - 3 * (ny + 1) - 1;
					if ((nx_index + nx) < 0 || (nx_index + nx) >= resolution || (ny_index + ny) < 0 || (ny_index + ny) >= resolution || (nz_index + nz) < 0 || (nz_index + nz) >= resolution) continue;

					OcKey morton_now = morton3D_64_Encode_magicbits(nx_index + nx, ny_index + ny, nz_index + nz);

					if (MSTG_HashAdd(((morton_now << (3 * (levelmax - tlevel) + 4)) | tlevel), nodes_buf) == true)
						delta(buffer2, nx_index + nx, ny_index + ny, nz_index + nz, nx_index >> 1, ny_index >> 1, nz_index >> 1, morton_now & 7);
				}
			}
			buffer1 = buffer2;
			buffer2 = 0;
			tlevel--;
		}

		nx_index = nx_index >> 1;
		ny_index = ny_index >> 1;
		nz_index = nz_index >> 1;
		resolution = resolution >> 1;
		for (int index = 0; index < 27; index++)
		{
			if (((buffer1 >> index) & 1) == 1)
			{
				int nz = index / 9 - 1;
				int ny = (index - 9 * (nz + 1)) / 3 - 1;;
				int nx = index - 9 * (nz + 1) - 3 * (ny + 1) - 1;
				if ((nx_index + nx) < 0 || (nx_index + nx) >= resolution || (ny_index + ny) < 0 || (ny_index + ny) >= resolution || (nz_index + nz) < 0 || (nz_index + nz) >= resolution) continue;

				OcKey morton_now = morton3D_64_Encode_magicbits(nx_index + nx, ny_index + ny, nz_index + nz);
				MSTG_HashAdd(((morton_now << (3 * (levelmax - tlevel) + 4)) | tlevel), nodes_buf);
			}
		}
	}

	__global__ void MSTG_ConstructNodesBufferImproveOriginal(
		DArray<OcKey> nodes_buf,
		DArray<OcKey> finest_nodes,
		Level levelmin,
		Level levelmax,
		int octreeType)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= ((levelmax - levelmin) * finest_nodes.size())) return;

		Level level_offset = tId / (finest_nodes.size());
		int index = tId - level_offset * (finest_nodes.size());
		if (index > 0 && ((finest_nodes[index] >> 3) == (finest_nodes[index - 1] >> 3))) return;
		OcKey nmorton = finest_nodes[index] >> ((level_offset + 1) * 3);
		Level nlevel = levelmax - (level_offset + 1);

		//Calculate the flip object one level up
		auto delta = [&](int& _buffer, int _mark) -> void {
			int _proximal_nx = 0, _proximal_ny = 0, _proximal_nz = 0;
			int  _ni_nx = 0, _ni_ny = 0, _ni_nz = 0;
			int _mark_temp = (_mark & 1) + ((_mark >> 3) & 1);
			if (_mark_temp > 1)
			{
				_ni_nx = 0;
				_proximal_nx = 1;
			}
			else if (_mark_temp == 1)
			{
				_ni_nx = 2;
				_proximal_nx = 0;
			}
			else
				_ni_nx = 1;

			_mark_temp = ((_mark >> 1) & 1) + ((_mark >> 4) & 1);
			if (_mark_temp > 1)
			{
				_ni_ny = 0;
				_proximal_ny = 1;
			}
			else if (_mark_temp == 1)
			{
				_ni_ny = 2;
				_proximal_ny = 0;
			}
			else
				_ni_ny = 1;

			_mark_temp = ((_mark >> 2) & 1) + ((_mark >> 5) & 1);
			if (_mark_temp > 1)
			{
				_ni_nz = 0;
				_proximal_nz = 1;
			}
			else if (_mark_temp == 1)
			{
				_ni_nz = 2;
				_proximal_nz = 0;
			}
			else
				_ni_nz = 1;

			_buffer = _buffer | (1 << (_proximal_nz * 9 + _proximal_ny * 3 + _ni_nx));
			_buffer = _buffer | (1 << (_proximal_nz * 9 + _ni_ny * 3 + _proximal_nx));
			_buffer = _buffer | (1 << (_ni_nz * 9 + _proximal_ny * 3 + _proximal_nx));
			if (octreeType < 2)
			{
				_buffer = _buffer | (1 << (_proximal_nz * 9 + _ni_ny * 3 + _ni_nx));
				_buffer = _buffer | (1 << (_ni_nz * 9 + _ni_ny * 3 + _proximal_nx));
				_buffer = _buffer | (1 << (_ni_nz * 9 + _proximal_ny * 3 + _ni_nx));
			}
			if (octreeType == 0)
				_buffer = _buffer | (1 << (_ni_nz * 9 + _ni_ny * 3 + _ni_nx));
			};

		//Update mask
		auto alpha = [&](OcKey& _nsign, OcKey& _mask1, OcKey& _mask2, int _ni) -> void {
			if (((_nsign >> _ni) & 1) == 0)
			{
				_mask1 = _mask1 | _mask2;
				_mask2 = (1 << _ni);
				return;
			}
			_mask2 = _mask1 = (1 << _ni);
			_ni = _ni + 3;
			while (_ni < 3 * nlevel)
			{
				_mask2 = _mask2 | (1 << _ni);
				if (((_nsign >> _ni) & 1) == 1) return;
				_ni = _ni + 3;
			}
			_mask2 = _mask1;
			return;
			};

		//octree: non-graded
		if (octreeType == 3)
			MSTG_HashAdd(((nmorton << (3 * (levelmax - nlevel) + 4)) | nlevel), nodes_buf);
		else
		{//octree: graded 
			if (nlevel == levelmin)
			{
				MSTG_HashAdd(((nmorton << (3 * (levelmax - levelmin) + 4)) | levelmin), nodes_buf);
				return;
			}

			MSTG_HashAdd(((nmorton << (3 * (levelmax - nlevel) + 4)) | nlevel), nodes_buf);

			OcKey nsign = (nmorton ^ (nmorton << 3));
			bool nx_gate = true, ny_gate = true, nz_gate = true;
			index = 6;
			OcKey nx_mask1 = 8, ny_mask1 = 16, nz_mask1 = 32;
			while ((index < 3 * nlevel) && (nx_gate || ny_gate || nz_gate))
			{
				if (nx_gate)
				{
					nx_mask1 = nx_mask1 | (1 << index);
					nx_gate = nx_gate ^ ((nsign >> index) & 1);
				}
				index++;
				if (ny_gate)
				{
					ny_mask1 = ny_mask1 | (1 << index);
					ny_gate = ny_gate ^ ((nsign >> index) & 1);
				}
				index++;
				if (nz_gate)
				{
					nz_mask1 = nz_mask1 | (1 << index);
					nz_gate = nz_gate ^ ((nsign >> index) & 1);
				}
				index++;
			}
			OcKey nx_mask2 = 8, ny_mask2 = 16, nz_mask2 = 32;
			if (nx_gate)
				nx_mask1 = nx_mask2;
			else if (((nsign >> 3) & 1) == 1)
			{
				nx_mask2 = nx_mask1;
				nx_mask1 = 8;
			}
			if (ny_gate)
				ny_mask1 = ny_mask2;
			else if (((nsign >> 4) & 1) == 1)
			{
				ny_mask2 = ny_mask1;
				ny_mask1 = 16;
			}
			if (nz_gate)
				nz_mask1 = nz_mask2;
			else if (((nsign >> 5) & 1) == 1)
			{
				nz_mask2 = nz_mask1;
				nz_mask1 = 32;
			}

			int buffer1 = 0, buffer2 = 0;
			Level tlevel = nlevel - 1;
			delta(buffer1, 0);
			while (tlevel > levelmin)
			{
				for (index = 1; index < 27; index++)
				{
					if (((buffer1 >> index) & 1) == 1)
					{
						int nz = index / 9;
						int ny = (index - 9 * nz) / 3;
						int nx = index - 9 * nz - 3 * ny;

						OcKey morton_now = nmorton;
						if (nx == 1) morton_now = morton_now ^ nx_mask1;
						else if (nx == 2) morton_now = morton_now ^ nx_mask2;
						if (ny == 1) morton_now = morton_now ^ ny_mask1;
						else if (ny == 2) morton_now = morton_now ^ ny_mask2;
						if (nz == 1) morton_now = morton_now ^ nz_mask1;
						else if (nz == 2) morton_now = morton_now ^ nz_mask2;
						if (MSTG_HashAdd((((morton_now >> 3 * (nlevel - tlevel)) << (3 * (levelmax - tlevel) + 4)) | tlevel), nodes_buf) == true)
							delta(buffer2, (((morton_now ^ nmorton) >> 3 * (nlevel - tlevel)) & 63));
					}
				}
				buffer1 = buffer2;
				buffer2 = 0;
				tlevel--;
				alpha(nsign, nx_mask1, nx_mask2, 3 * (nlevel - tlevel));
				alpha(nsign, ny_mask1, ny_mask2, 3 * (nlevel - tlevel) + 1);
				alpha(nsign, nz_mask1, nz_mask2, 3 * (nlevel - tlevel) + 2);
			}

			for (index = 1; index < 27; index++)
			{
				if (((buffer1 >> index) & 1) == 1)
				{
					int nz = index / 9;
					int ny = (index - 9 * nz) / 3;
					int nx = index - 9 * nz - 3 * ny;

					OcKey morton_now = nmorton;
					if (nx == 1) morton_now = morton_now ^ nx_mask1;
					else if (nx == 2) morton_now = morton_now ^ nx_mask2;
					if (ny == 1) morton_now = morton_now ^ ny_mask1;
					else if (ny == 2) morton_now = morton_now ^ ny_mask2;
					if (nz == 1) morton_now = morton_now ^ nz_mask1;
					else if (nz == 2) morton_now = morton_now ^ nz_mask2;
					MSTG_HashAdd((((morton_now >> 3 * (nlevel - tlevel)) << (3 * (levelmax - tlevel) + 4)) | tlevel), nodes_buf);
				}
			}
		}
	}


	__global__ void MSTG_ConstructNodesBufferImprove(
		DArray<OcKey> nodes_buf,
		DArray<OcKey> finest_nodes,
		Level levelmin,
		Level levelmax,
		int octreeType)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= ((levelmax - levelmin) * finest_nodes.size())) return;

		Level level_offset = tId / (finest_nodes.size());
		int index = tId - level_offset * (finest_nodes.size());
		if (index > 0 && ((finest_nodes[index] >> 3) == (finest_nodes[index - 1] >> 3))) return;
		OcKey nmorton = finest_nodes[index] >> ((level_offset + 1) * 3);
		Level nlevel = levelmax - (level_offset + 1);

		//Calculate the flip object one level up
		auto delta = [&](int& _buffer, int _ni_nx, int _ni_ny, int _ni_nz, int _mark) -> void {
			bool _nx = true, _ny = true, _nz = true;
			bool _proximal_nx = false, _proximal_ny = false, _proximal_nz = false;

			int _mark_temp = (_mark & 1) + ((_mark >> 3) & 1);
			if (_mark_temp > 1)
			{
				_nx = false;
				_ni_nx = 1;
			}
			else if (_mark_temp == 1)
			{
				_proximal_nx = true;
				_ni_nx = 0;
			}
			_mark_temp = ((_mark >> 1) & 1) + ((_mark >> 4) & 1);
			if (_mark_temp > 1)
			{
				_ny = false;
				_ni_ny = 1;
			}
			else if (_mark_temp == 1)
			{
				_proximal_ny = true;
				_ni_ny = 0;
			}
			_mark_temp = ((_mark >> 2) & 1) + ((_mark >> 5) & 1);
			if (_mark_temp > 1)
			{
				_nz = false;
				_ni_nz = 1;
			}
			else if (_mark_temp == 1)
			{
				_proximal_nz = true;
				_ni_nz = 0;
			}

			if (_proximal_nx || _proximal_ny || _proximal_nz == true)
			{
				if (_proximal_nx == true)
					_buffer = _buffer | (1 << (_ni_nz * 9 + _ni_ny * 3 + 2));
				if (_proximal_ny == true)
					_buffer = _buffer | (1 << (_ni_nz * 9 + 2 * 3 + _ni_nx));
				if (_proximal_nz == true)
					_buffer = _buffer | (1 << (2 * 9 + _ni_ny * 3 + _ni_nx));

				if (octreeType < 2)
				{
					if (_nx == true && _ny == true)
						_buffer = _buffer | (1 << (_ni_nz * 9 + (int(_ny) + int(_proximal_ny)) * 3 + (int(_nx) + int(_proximal_nx))));
					if (_nx == true && _nz == true)
						_buffer = _buffer | (1 << ((int(_nz) + int(_proximal_nz)) * 9 + _ni_ny * 3 + (int(_nx) + int(_proximal_nx))));
					if (_nz == true && _ny == true)
						_buffer = _buffer | (1 << ((int(_nz) + int(_proximal_nz)) * 9 + (int(_ny) + int(_proximal_ny)) * 3 + _ni_nx));
				}

				if (octreeType == 0)
				{
					if (_nx == true && _ny == true && _nz == true)
						_buffer = _buffer | (1 << ((int(_nz) + int(_proximal_nz)) * 9 + (int(_ny) + int(_proximal_ny)) * 3 + (int(_nx) + int(_proximal_nx))));
				}
			}
			else
			{
				if (_nx == true)
					_buffer = _buffer | (1 << (_ni_nz * 9 + _ni_ny * 3 + 1));
				if (_ny == true)
					_buffer = _buffer | (1 << (_ni_nz * 9 + 1 * 3 + _ni_nx));
				if (_nz == true)
					_buffer = _buffer | (1 << (1 * 9 + _ni_ny * 3 + _ni_nx));

				if (octreeType < 2)
				{
					if (_nx == true && _ny == true)
						_buffer = _buffer | (1 << (_ni_nz * 9 + 1 * 3 + 1));
					if (_nx == true && _nz == true)
						_buffer = _buffer | (1 << (1 * 9 + _ni_ny * 3 + 1));
					if (_nz == true && _ny == true)
						_buffer = _buffer | (1 << (1 * 9 + 1 * 3 + _ni_nx));
				}

				if (octreeType == 0)
				{
					if (_nx == true && _ny == true && _nz == true)
						_buffer = _buffer | (1 << (1 * 9 + 1 * 3 + 1));
				}
			}
			};

		//Update mask
		auto alpha = [&](OcKey& _nsign, OcKey& _mask1, OcKey& _mask2, int _ni) -> void {
			if (((_nsign >> _ni) & 1) == 0)
			{
				_mask1 = _mask1 | _mask2;
				_mask2 = (1 << _ni);
				return;
			}
			_mask2 = _mask1 = (1 << _ni);
			_ni = _ni + 3;
			while (_ni < 3 * nlevel)
			{
				_mask2 = _mask2 | (1 << _ni);
				if (((_nsign >> _ni) & 1) == 1) return;
				_ni = _ni + 3;
			}
			_mask2 = _mask1;
			return;
			};

		//octree: non-graded
		if (octreeType == 3)
			MSTG_HashAdd(((nmorton << (3 * (levelmax - nlevel) + 4)) | nlevel), nodes_buf);
		else
		{//octree: graded 
			if (nlevel == levelmin)
			{
				MSTG_HashAdd(((nmorton << (3 * (levelmax - levelmin) + 4)) | levelmin), nodes_buf);
				return;
			}

			MSTG_HashAdd(((nmorton << (3 * (levelmax - nlevel) + 4)) | nlevel), nodes_buf);

			OcKey nsign = (nmorton ^ (nmorton << 3));
			bool nx_gate = true, ny_gate = true, nz_gate = true;
			index = 6;
			OcKey nx_mask1 = 8, ny_mask1 = 16, nz_mask1 = 32;
			while ((index < 3 * nlevel) && (nx_gate || ny_gate || nz_gate))
			{
				if (nx_gate)
				{
					nx_mask1 = nx_mask1 | (1 << index);
					nx_gate = nx_gate ^ ((nsign >> index) & 1);
				}
				index++;
				if (ny_gate)
				{
					ny_mask1 = ny_mask1 | (1 << index);
					ny_gate = ny_gate ^ ((nsign >> index) & 1);
				}
				index++;
				if (nz_gate)
				{
					nz_mask1 = nz_mask1 | (1 << index);
					nz_gate = nz_gate ^ ((nsign >> index) & 1);
				}
				index++;
			}
			OcKey nx_mask2 = 8, ny_mask2 = 16, nz_mask2 = 32;
			if (nx_gate)
				nx_mask1 = nx_mask2;
			else if (((nsign >> 3) & 1) == 1)
			{
				nx_mask2 = nx_mask1;
				nx_mask1 = 8;
			}
			if (ny_gate)
				ny_mask1 = ny_mask2;
			else if (((nsign >> 4) & 1) == 1)
			{
				ny_mask2 = ny_mask1;
				ny_mask1 = 16;
			}
			if (nz_gate)
				nz_mask1 = nz_mask2;
			else if (((nsign >> 5) & 1) == 1)
			{
				nz_mask2 = nz_mask1;
				nz_mask1 = 32;
			}

			int buffer1 = 0, buffer2 = 0;
			Level tlevel = nlevel - 1;
			delta(buffer1, 0, 0, 0, 0);
			while (tlevel > levelmin)
			{
				for (index = 1; index < 27; index++)
				{
					if (((buffer1 >> index) & 1) == 1)
					{
						int nz = index / 9;
						int ny = (index - 9 * nz) / 3;
						int nx = index - 9 * nz - 3 * ny;

						OcKey morton_now = nmorton;
						if (nx == 1) morton_now = morton_now ^ nx_mask1;
						else if (nx == 2) morton_now = morton_now ^ nx_mask2;
						if (ny == 1) morton_now = morton_now ^ ny_mask1;
						else if (ny == 2) morton_now = morton_now ^ ny_mask2;
						if (nz == 1) morton_now = morton_now ^ nz_mask1;
						else if (nz == 2) morton_now = morton_now ^ nz_mask2;
						MSTG_HashAdd((((morton_now >> 3 * (nlevel - tlevel)) << (3 * (levelmax - tlevel) + 4)) | tlevel), nodes_buf);
						delta(buffer2, nx, ny, nz, (((morton_now ^ nmorton) >> 3 * (nlevel - tlevel)) & 63));
					}
				}
				buffer1 = buffer2;
				buffer2 = 0;
				tlevel--;
				alpha(nsign, nx_mask1, nx_mask2, 3 * (nlevel - tlevel));
				alpha(nsign, ny_mask1, ny_mask2, 3 * (nlevel - tlevel) + 1);
				alpha(nsign, nz_mask1, nz_mask2, 3 * (nlevel - tlevel) + 2);
			}

			for (index = 1; index < 27; index++)
			{
				if (((buffer1 >> index) & 1) == 1)
				{
					int nz = index / 9;
					int ny = (index - 9 * nz) / 3;
					int nx = index - 9 * nz - 3 * ny;

					OcKey morton_now = nmorton;
					if (nx == 1) morton_now = morton_now ^ nx_mask1;
					else if (nx == 2) morton_now = morton_now ^ nx_mask2;
					if (ny == 1) morton_now = morton_now ^ ny_mask1;
					else if (ny == 2) morton_now = morton_now ^ ny_mask2;
					if (nz == 1) morton_now = morton_now ^ nz_mask1;
					else if (nz == 2) morton_now = morton_now ^ nz_mask2;
					MSTG_HashAdd((((morton_now >> 3 * (nlevel - tlevel)) << (3 * (levelmax - tlevel) + 4)) | tlevel), nodes_buf);
				}
			}
		}
	}

	__global__ void MSTG_CountNodesBuffer(
		DArray<int> mark,
		DArray<OcKey> nodes_buf)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes_buf.size()) return;

		if (nodes_buf[tId] != (OcKey)0)
			mark[tId] = 1;
	}

	__global__ void MSTG_FetchNodesNorepeat(
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

	template <typename Real, typename Coord>
	__global__ void MSTG_ComputeNodesAll(
		DArray<AdaptiveGridNode> nodes,
		DArray<OcKey> nodes_morton,
		Level levelmin,
		Level levelmax,
		Coord origin,
		Real dx,
		int minnum)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= (nodes_morton.size() + minnum)) return;

		if (tId < minnum)
		{
			OcIndex gnx, gny, gnz;
			morton3D_64_Decode_magicbits(OcKey(tId), gnx, gny, gnz);
			Real gdx = dx * (1 << (levelmax - levelmin));
			Coord gpos(origin[0] + (gnx + 0.5)*gdx, origin[1] + (gny + 0.5)*gdx, origin[2] + (gnz + 0.5)*gdx);

			nodes[tId] = AdaptiveGridNode(levelmin, OcKey(tId), gpos);
		}
		else
		{
			int index = tId - minnum;
			Level nlevel = nodes_morton[index] & 15;
			OcKey nmorton = (nodes_morton[index] >> (3 * (levelmax - nlevel) + 4)) << 3;
			nlevel++;

			OcIndex gnx, gny, gnz;
			morton3D_64_Decode_magicbits(nmorton, gnx, gny, gnz);
			Real gdx = dx * (1 << (levelmax - nlevel));
			Coord gpos(origin[0] + (gnx + 0.5) * gdx, origin[1] + (gny + 0.5) * gdx, origin[2] + (gnz + 0.5) * gdx);

			nodes[minnum + 8 * index] = AdaptiveGridNode(nlevel, nmorton, gpos);
			nodes[minnum + 8 * index + 1] = AdaptiveGridNode(nlevel, nmorton + 1, gpos + Coord(gdx, 0, 0));
			nodes[minnum + 8 * index + 2] = AdaptiveGridNode(nlevel, nmorton + 2, gpos + Coord(0, gdx, 0));
			nodes[minnum + 8 * index + 3] = AdaptiveGridNode(nlevel, nmorton + 3, gpos + Coord(gdx, gdx, 0));
			nodes[minnum + 8 * index + 4] = AdaptiveGridNode(nlevel, nmorton + 4, gpos + Coord(0, 0, gdx));
			nodes[minnum + 8 * index + 5] = AdaptiveGridNode(nlevel, nmorton + 5, gpos + Coord(gdx, 0, gdx));
			nodes[minnum + 8 * index + 6] = AdaptiveGridNode(nlevel, nmorton + 6, gpos + Coord(0, gdx, gdx));
			nodes[minnum + 8 * index + 7] = AdaptiveGridNode(nlevel, nmorton + 7, gpos + Coord(gdx, gdx, gdx));
		}
	}

	__global__ void MSTG_ComputeNodesChildRelationships(
		DArray<AdaptiveGridNode> nodes,
		DArray<OcKey> nodes_buf,
		Level levelmin,
		Level levelmax,
		int minnum)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= (nodes_buf.size())) return;

		//OcKey nmorton = nodes_buf[tId];
		//Calculate the length of the longest common prefix between tId and j
		auto delta = [&](int _j) -> int {
			if (_j < 0 || _j >= nodes_buf.size()) return -1;
			return __clzll(nodes_buf[tId] ^ nodes_buf[_j]);
		};

		int delta_min = delta(tId + 1);
		delta_min = ((int)(delta_min / 3)) * 3;
		if (delta_min < (64 - 3 * (levelmax - levelmin) - 4))
		{
			int find = nodes_buf[tId] >> (3 * (levelmax - levelmin) + 4);
			nodes[find].m_fchild = minnum + 8 * tId;
			return;
		}

		// Find the other end using binary search
		int len_max = 2;
		while (delta(tId + len_max) >= delta_min)
		{
			len_max *= 2;
		}

		int len = 0;
		for (int t = len_max / 2; t > 0; t = t / 2)
		{
			if (delta(tId + (len + t)) >= delta_min)
			{
				len = len + t;
			}
		}
		int find = tId + len;

		Level nlevel = nodes_buf[tId] & 15;
		find -= (nlevel - 1 - (nodes_buf[find] & 15));

		int find_off = (nodes_buf[tId] >> (3 * (levelmax - nlevel) + 4)) & 7;
		nodes[minnum + 8 * find + find_off].m_fchild = minnum + 8 * tId;
	}

	__global__ void MSTG_ComputeNodesChildRelationshipsSecond(
		DArray<AdaptiveGridNode> nodes,
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

		Level nlevel = nodes_buf[tId] & 15;
		OcKey nmorton = nodes_buf[tId] >> (3 * (levelmax - nlevel) + 4);

		if (nlevel == levelmin)
		{
			nodes[nmorton].m_fchild = minnum + 8 * tId;
			return;
		}

		OcKey pmorton = ((nmorton >> 3) << (3 * (levelmax - nlevel + 1) + 4)) | (nlevel - 1);
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

		int find_off = nmorton & 7;
		nodes[minnum + 8 * find + find_off].m_fchild = minnum + 8 * tId;
	}

	template<typename TDataType>
	void MSTsGeneratorHelper<TDataType>::ConstructionFromScratch(
		std::shared_ptr<AdaptiveGridSet<TDataType>> AGridSet,
		DArray<OcKey>& mSeed,
		Level mLevelnum,
		int mType)
	{
		auto& nodes = AGridSet->getAGrids();
		Real m_dx = AGridSet->getDx();
		Coord m_origin = AGridSet->getOrigin();
		Level m_levelmax = AGridSet->getLevelMax();
		AGridSet->setLevelNum(mLevelnum);
		assert(mLevelnum <= m_levelmax);
		AGridSet->setOctreeType(mType);

		int max_resolution = (1 << m_levelmax);
		int buf_num = 3 * (mSeed.size());
		buf_num += (buf_num % 2 == 0) ? 1 : 0;
		DArray<OcKey> buffer_key(buf_num);
		buffer_key.reset();

		cuExecute(mSeed.size(),
			MSTG_ConstructNodesBufferIndexSingle,
			buffer_key,
			mSeed,
			m_levelmax - mLevelnum + 1,
			m_levelmax,
			mType,
			max_resolution);

		DArray<int> buffer_mark(buf_num);
		buffer_mark.reset();
		cuExecute(buf_num,
			MSTG_CountNodesBuffer,
			buffer_mark,
			buffer_key);

		Reduction<int> reduce;
		int fnode_num = reduce.accumulate(buffer_mark.begin(), buffer_mark.size());

		Scan<int> scan;
		scan.exclusive(buffer_mark.begin(), buffer_mark.size());

		DArray<OcKey> nodes_morton(fnode_num);
		cuExecute(buf_num,
			MSTG_FetchNodesNorepeat,
			nodes_morton,
			buffer_mark,
			buffer_key);

		thrust::sort(thrust::device, nodes_morton.begin(), nodes_morton.begin() + nodes_morton.size(), thrust::greater<OcKey>());

		int min_resolution = (1 << (m_levelmax - mLevelnum + 1));
		int min_nodes_num = min_resolution * min_resolution*min_resolution;
		nodes.resize(min_nodes_num + 8 * fnode_num);
		cuExecute(min_nodes_num + fnode_num,
			MSTG_ComputeNodesAll,
			nodes,
			nodes_morton,
			m_levelmax - mLevelnum + 1,
			m_levelmax,
			m_origin,
			m_dx,
			min_nodes_num);

		cuExecute(fnode_num,
			MSTG_ComputeNodesChildRelationshipsSecond,
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
	__global__ void DU_PutInternalNodeAndCount(
		DArray<OcKey> nodes_buf,
		DArray<int> buf_count,
		DArray<AdaptiveGridNode> nodes,
		AdaptiveGridSet<TDataType> gridSet,
		Level levelmax,
		int octreeType)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes.size()) return;
		if (nodes[tId].isLeaf()) return;

		OcKey nmorton = nodes[tId].m_morton;
		Level nlevel = nodes[tId].m_level;
		if (nlevel == (levelmax - 1))
		{
			MSTG_HashAddAndCount(((nmorton << (3 * (levelmax - nlevel) + 4)) | nlevel), 1, nodes_buf, buf_count);
			return;
		}

		int ncount = 0;
		int cindex = nodes[tId].m_fchild;
		for (int i = 0; i < 8; i++)
		{
			if (!nodes[cindex + i].isLeaf())
				ncount++;
		}
		if (octreeType == 3)
		{
			MSTG_HashAddAndCount(((nmorton << (3 * (levelmax - nlevel) + 4)) | nlevel), ncount, nodes_buf, buf_count);
			return;
		}

		OcIndex nx_index, ny_index, nz_index;
		RecoverFromMortonCode(nmorton, nx_index, ny_index, nz_index);
		int vindex = 32;
		if (octreeType == 1) vindex = 56;
		else if (octreeType == 0) vindex = 64;
		for (int i = 8; i < vindex; i++)
		{
			OcIndex nx_temp = 2 * nx_index + verification[i][0];
			OcIndex ny_temp = 2 * ny_index + verification[i][1];
			OcIndex nz_temp = 2 * nz_index + verification[i][2];
			int resolution = 1 << (nlevel + 1);
			if (nx_temp < 0 || nx_temp >= resolution || ny_temp < 0 || ny_temp >= resolution || nz_temp < 0 || nz_temp >= resolution) continue;

			OcKey morton_temp = CalculateMortonCode(nx_temp, ny_temp, nz_temp);
			if (!(gridSet.accessRandom(cindex, morton_temp, nlevel + 1)))
				ncount++;
		}

		MSTG_HashAddAndCount(((nmorton << (3 * (levelmax - nlevel) + 4)) | nlevel), ncount, nodes_buf, buf_count);
		return;
	}

	__global__ void DU_ConstructNodesBufferDecrease(
		DArray<OcKey> nodes_buf,
		DArray<int> buf_count,
		DArray<OcKey> decrease_nodes,
		Level levelmin,
		Level levelmax,
		int octreeType,
		int resolution)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= decrease_nodes.size()) return;
		if (tId > 0 && ((decrease_nodes[tId] >> 3) == (decrease_nodes[tId - 1] >> 3))) return;

		Level nlevel = levelmax - 1;
		OcKey nmorton = decrease_nodes[tId] >> 3;
		resolution = resolution >> 1;

		//Calculate the flip object one level up
		auto delta = [&](int& _buffer, int _nx, int _ny, int _nz, int _start_nx, int _start_ny, int _start_nz, Level le) -> void {
			int _tnx, _tny, _tnz;
			if ((_nx & 1) == 0)
				_tnx = (_nx >> 1) - 1;
			else
				_tnx = (_nx >> 1) + 1;
			if ((_ny & 1) == 0)
				_tny = (_ny >> 1) - 1;
			else
				_tny = (_ny >> 1) + 1;
			if ((_nz & 1) == 0)
				_tnz = (_nz >> 1) - 1;
			else
				_tnz = (_nz >> 1) + 1;

			int _off_nx = _tnx - _start_nx;
			int _off_ny = _tny - _start_ny;
			int _off_nz = _tnz - _start_nz;
			int _fix_nx = (_nx >> 1) - _start_nx;
			int _fix_ny = (_ny >> 1) - _start_ny;
			int _fix_nz = (_nz >> 1) - _start_nz;

			if (MSTG_HashCountSub((((CalculateMortonCode(_start_nx + _fix_nx, _start_ny + _fix_ny, _start_nz + _fix_nz)) << (3 * (levelmax - le) + 4)) | le), nodes_buf, buf_count) == 0)
				_buffer = _buffer | (1 << ((_fix_nz + 1) * 9 + (_fix_ny + 1) * 3 + (_fix_nx + 1)));
			if (octreeType < 3)
			{
				if (MSTG_HashCountSub((((CalculateMortonCode(_start_nx + _off_nx, _start_ny + _fix_ny, _start_nz + _fix_nz)) << (3 * (levelmax - le) + 4)) | le), nodes_buf, buf_count) == 0)
					_buffer = _buffer | (1 << ((_fix_nz + 1) * 9 + (_fix_ny + 1) * 3 + (_off_nx + 1)));
				if (MSTG_HashCountSub((((CalculateMortonCode(_start_nx + _fix_nx, _start_ny + _off_ny, _start_nz + _fix_nz)) << (3 * (levelmax - le) + 4)) | le), nodes_buf, buf_count) == 0)
					_buffer = _buffer | (1 << ((_fix_nz + 1) * 9 + (_off_ny + 1) * 3 + (_fix_nx + 1)));
				if (MSTG_HashCountSub((((CalculateMortonCode(_start_nx + _fix_nx, _start_ny + _fix_ny, _start_nz + _off_nz)) << (3 * (levelmax - le) + 4)) | le), nodes_buf, buf_count) == 0)
					_buffer = _buffer | (1 << ((_off_nz + 1) * 9 + (_fix_ny + 1) * 3 + (_fix_nx + 1)));
			}
			if (octreeType < 2)
			{
				if (MSTG_HashCountSub((((CalculateMortonCode(_start_nx + _off_nx, _start_ny + _off_ny, _start_nz + _fix_nz)) << (3 * (levelmax - le) + 4)) | le), nodes_buf, buf_count) == 0)
					_buffer = _buffer | (1 << ((_fix_nz + 1) * 9 + (_off_ny + 1) * 3 + (_off_nx + 1)));
				if (MSTG_HashCountSub((((CalculateMortonCode(_start_nx + _fix_nx, _start_ny + _off_ny, _start_nz + _off_nz)) << (3 * (levelmax - le) + 4)) | le), nodes_buf, buf_count) == 0)
					_buffer = _buffer | (1 << ((_off_nz + 1) * 9 + (_off_ny + 1) * 3 + (_fix_nx + 1)));
				if (MSTG_HashCountSub((((CalculateMortonCode(_start_nx + _off_nx, _start_ny + _fix_ny, _start_nz + _off_nz)) << (3 * (levelmax - le) + 4)) | le), nodes_buf, buf_count) == 0)
					_buffer = _buffer | (1 << ((_off_nz + 1) * 9 + (_fix_ny + 1) * 3 + (_off_nx + 1)));
			}
			if (octreeType == 0)
				if (MSTG_HashCountSub((((CalculateMortonCode(_start_nx + _off_nx, _start_ny + _off_ny, _start_nz + _off_nz)) << (3 * (levelmax - le) + 4)) | le), nodes_buf, buf_count) == 0)
					_buffer = _buffer | (1 << ((_off_nz + 1) * 9 + (_off_ny + 1) * 3 + (_off_nx + 1)));
			};

		if (nlevel == levelmin)
		{
			MSTG_HashCountSub(((nmorton << (3 * (levelmax - levelmin) + 4)) | levelmin), nodes_buf, buf_count);
			return;
		}

		int buffer1 = 0, buffer2 = 0;
		OcIndex nx_index, ny_index, nz_index;
		RecoverFromMortonCode(nmorton, nx_index, ny_index, nz_index);
		if (MSTG_HashCountSub(((nmorton << (3 * (levelmax - nlevel) + 4)) | nlevel), nodes_buf, buf_count) == 0)
			delta(buffer1, nx_index, ny_index, nz_index, nx_index >> 1, ny_index >> 1, nz_index >> 1, nlevel - 1);
		Level tlevel = nlevel - 1;
		while (tlevel > levelmin)
		{
			nx_index = nx_index >> 1;
			ny_index = ny_index >> 1;
			nz_index = nz_index >> 1;
			resolution = resolution >> 1;

			for (int index = 0; index < 27; index++)
			{
				if (((buffer1 >> index) & 1) == 1)
				{
					int nz = index / 9 - 1;
					int ny = (index - 9 * (nz + 1)) / 3 - 1;;
					int nx = index - 9 * (nz + 1) - 3 * (ny + 1) - 1;
					if ((nx_index + nx) < 0 || (nx_index + nx) >= resolution || (ny_index + ny) < 0 || (ny_index + ny) >= resolution || (nz_index + nz) < 0 || (nz_index + nz) >= resolution) continue;

					delta(buffer2, nx_index + nx, ny_index + ny, nz_index + nz, nx_index >> 1, ny_index >> 1, nz_index >> 1, tlevel - 1);
				}
			}
			buffer1 = buffer2;
			buffer2 = 0;
			tlevel--;
		}
	}

	__global__ void DU_ConstructNodesBufferIncrease(
		DArray<OcKey> nodes_buf,
		DArray<int> buf_count,
		DArray<OcKey> finest_nodes,
		Level levelmin,
		Level levelmax,
		int octreeType,
		int resolution)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= ((levelmax - levelmin) * finest_nodes.size())) return;

		Level level_offset = tId / (finest_nodes.size());
		int index = tId - level_offset * (finest_nodes.size());
		if (index > 0 && ((finest_nodes[index] >> ((level_offset + 1) * 3)) == (finest_nodes[index - 1] >> ((level_offset + 1) * 3)))) return;
		OcKey nmorton = finest_nodes[index] >> ((level_offset + 1) * 3);
		Level nlevel = levelmax - (level_offset + 1);
		resolution = resolution >> (level_offset + 1);

		//Calculate the flip object one level up
		auto delta = [&](int& _buffer, int _nx, int _ny, int _nz, int _start_nx, int _start_ny, int _start_nz, int _mark) -> void {
			int _tnx, _tny, _tnz;
			if ((_mark & 1) == 0)
				_tnx = (_nx >> 1) - 1;
			else
				_tnx = (_nx >> 1) + 1;
			if ((_mark & 2) == 0)
				_tny = (_ny >> 1) - 1;
			else
				_tny = (_ny >> 1) + 1;
			if ((_mark & 4) == 0)
				_tnz = (_nz >> 1) - 1;
			else
				_tnz = (_nz >> 1) + 1;

			int _off_nx = _tnx - _start_nx;
			int _off_ny = _tny - _start_ny;
			int _off_nz = _tnz - _start_nz;
			int _fix_nx = (_nx >> 1) - _start_nx;
			int _fix_ny = (_ny >> 1) - _start_ny;
			int _fix_nz = (_nz >> 1) - _start_nz;

			_buffer = _buffer | (1 << ((_fix_nz + 1) * 9 + (_fix_ny + 1) * 3 + (_off_nx + 1)));
			_buffer = _buffer | (1 << ((_fix_nz + 1) * 9 + (_off_ny + 1) * 3 + (_fix_nx + 1)));
			_buffer = _buffer | (1 << ((_off_nz + 1) * 9 + (_fix_ny + 1) * 3 + (_fix_nx + 1)));
			if (octreeType < 2)
			{
				_buffer = _buffer | (1 << ((_fix_nz + 1) * 9 + (_off_ny + 1) * 3 + (_off_nx + 1)));
				_buffer = _buffer | (1 << ((_off_nz + 1) * 9 + (_off_ny + 1) * 3 + (_fix_nx + 1)));
				_buffer = _buffer | (1 << ((_off_nz + 1) * 9 + (_fix_ny + 1) * 3 + (_off_nx + 1)));
			}
			if (octreeType == 0)
				_buffer = _buffer | (1 << ((_off_nz + 1) * 9 + (_off_ny + 1) * 3 + (_off_nx + 1)));
			};

		if (nlevel == levelmin)
		{
			MSTG_HashAddCount(((nmorton << (3 * (levelmax - levelmin) + 4)) | levelmin), nodes_buf, buf_count);
			return;
		}

		//octree: non-graded
		if (octreeType == 3)
			MSTG_HashAddCount(((nmorton << (3 * (levelmax - nlevel) + 4)) | nlevel), nodes_buf, buf_count);
		else
		{//octree: graded 
			MSTG_HashAddCount(((nmorton << (3 * (levelmax - nlevel) + 4)) | nlevel), nodes_buf, buf_count);

			int buffer1 = 0, buffer2 = 0;
			OcIndex nx_index, ny_index, nz_index;
			morton3D_64_Decode_magicbits(nmorton, nx_index, ny_index, nz_index);
			delta(buffer1, nx_index, ny_index, nz_index, nx_index >> 1, ny_index >> 1, nz_index >> 1, nmorton & 7);
			Level tlevel = nlevel - 1;
			while (tlevel > levelmin)
			{
				nx_index = nx_index >> 1;
				ny_index = ny_index >> 1;
				nz_index = nz_index >> 1;
				resolution = resolution >> 1;

				for (index = 0; index < 27; index++)
				{
					if (((buffer1 >> index) & 1) == 1)
					{
						int nz = index / 9 - 1;
						int ny = (index - 9 * (nz + 1)) / 3 - 1;;
						int nx = index - 9 * (nz + 1) - 3 * (ny + 1) - 1;
						if ((nx_index + nx) < 0 || (nx_index + nx) >= resolution || (ny_index + ny) < 0 || (ny_index + ny) >= resolution || (nz_index + nz) < 0 || (nz_index + nz) >= resolution) continue;

						OcKey morton_now = morton3D_64_Encode_magicbits(nx_index + nx, ny_index + ny, nz_index + nz);

						if (MSTG_HashAddCount(((morton_now << (3 * (levelmax - tlevel) + 4)) | tlevel), nodes_buf, buf_count) == true)
							delta(buffer2, nx_index + nx, ny_index + ny, nz_index + nz, nx_index >> 1, ny_index >> 1, nz_index >> 1, morton_now & 7);
					}
				}
				buffer1 = buffer2;
				buffer2 = 0;
				tlevel--;
			}

			nx_index = nx_index >> 1;
			ny_index = ny_index >> 1;
			nz_index = nz_index >> 1;
			resolution = resolution >> 1;
			for (index = 0; index < 27; index++)
			{
				if (((buffer1 >> index) & 1) == 1)
				{
					int nz = index / 9 - 1;
					int ny = (index - 9 * (nz + 1)) / 3 - 1;;
					int nx = index - 9 * (nz + 1) - 3 * (ny + 1) - 1;
					if ((nx_index + nx) < 0 || (nx_index + nx) >= resolution || (ny_index + ny) < 0 || (ny_index + ny) >= resolution || (nz_index + nz) < 0 || (nz_index + nz) >= resolution) continue;

					OcKey morton_now = morton3D_64_Encode_magicbits(nx_index + nx, ny_index + ny, nz_index + nz);
					MSTG_HashAddCount(((morton_now << (3 * (levelmax - tlevel) + 4)) | tlevel), nodes_buf, buf_count);
				}
			}
		}
	}

	__global__ void DU_ConstructNodesBufferIncreaseSingle(
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
		if (tId > 0 && ((finest_nodes[tId] >> 3) == (finest_nodes[tId - 1] >> 3))) return;

		OcKey nmorton = finest_nodes[tId] >> 3;
		Level nlevel = levelmax - 1;
		resolution = resolution >> 1;

		//Calculate the flip object one level up
		auto delta = [&](int& _buffer, int _nx, int _ny, int _nz, int _start_nx, int _start_ny, int _start_nz) -> void {
			int _tnx, _tny, _tnz;
			if ((_nx & 1) == 0)
				_tnx = (_nx >> 1) - 1;
			else
				_tnx = (_nx >> 1) + 1;
			if ((_ny & 1) == 0)
				_tny = (_ny >> 1) - 1;
			else
				_tny = (_ny >> 1) + 1;
			if ((_nz & 1) == 0)
				_tnz = (_nz >> 1) - 1;
			else
				_tnz = (_nz >> 1) + 1;

			int _off_nx = _tnx - _start_nx;
			int _off_ny = _tny - _start_ny;
			int _off_nz = _tnz - _start_nz;
			int _fix_nx = (_nx >> 1) - _start_nx;
			int _fix_ny = (_ny >> 1) - _start_ny;
			int _fix_nz = (_nz >> 1) - _start_nz;

			_buffer = _buffer | (1 << ((_fix_nz + 1) * 9 + (_fix_ny + 1) * 3 + (_fix_nx + 1)));
			if (octreeType < 3)
			{
				_buffer = _buffer | (1 << ((_fix_nz + 1) * 9 + (_fix_ny + 1) * 3 + (_off_nx + 1)));
				_buffer = _buffer | (1 << ((_fix_nz + 1) * 9 + (_off_ny + 1) * 3 + (_fix_nx + 1)));
				_buffer = _buffer | (1 << ((_off_nz + 1) * 9 + (_fix_ny + 1) * 3 + (_fix_nx + 1)));
			}
			if (octreeType < 2)
			{
				_buffer = _buffer | (1 << ((_fix_nz + 1) * 9 + (_off_ny + 1) * 3 + (_off_nx + 1)));
				_buffer = _buffer | (1 << ((_off_nz + 1) * 9 + (_off_ny + 1) * 3 + (_fix_nx + 1)));
				_buffer = _buffer | (1 << ((_off_nz + 1) * 9 + (_fix_ny + 1) * 3 + (_off_nx + 1)));
			}
			if (octreeType == 0)
				_buffer = _buffer | (1 << ((_off_nz + 1) * 9 + (_off_ny + 1) * 3 + (_off_nx + 1)));
			};

		if (nlevel == levelmin)
		{
			MSTG_HashAddCount(((nmorton << (3 * (levelmax - levelmin) + 4)) | levelmin), nodes_buf, buf_count);
			return;
		}

		MSTG_HashAddCount(((nmorton << (3 * (levelmax - nlevel) + 4)) | nlevel), nodes_buf, buf_count);

		int buffer1 = 0, buffer2 = 0;
		OcIndex nx_index, ny_index, nz_index;
		morton3D_64_Decode_magicbits(nmorton, nx_index, ny_index, nz_index);
		delta(buffer1, nx_index, ny_index, nz_index, nx_index >> 1, ny_index >> 1, nz_index >> 1);
		Level tlevel = nlevel - 1;
		while (tlevel > levelmin)
		{
			nx_index = nx_index >> 1;
			ny_index = ny_index >> 1;
			nz_index = nz_index >> 1;
			resolution = resolution >> 1;

			for (int index = 0; index < 27; index++)
			{
				if (((buffer1 >> index) & 1) == 1)
				{
					int nz = index / 9 - 1;
					int ny = (index - 9 * (nz + 1)) / 3 - 1;;
					int nx = index - 9 * (nz + 1) - 3 * (ny + 1) - 1;
					if ((nx_index + nx) < 0 || (nx_index + nx) >= resolution || (ny_index + ny) < 0 || (ny_index + ny) >= resolution || (nz_index + nz) < 0 || (nz_index + nz) >= resolution) continue;

					OcKey morton_now = morton3D_64_Encode_magicbits(nx_index + nx, ny_index + ny, nz_index + nz);

					if (MSTG_HashAddCount(((morton_now << (3 * (levelmax - tlevel) + 4)) | tlevel), nodes_buf, buf_count) == true)
						delta(buffer2, nx_index + nx, ny_index + ny, nz_index + nz, nx_index >> 1, ny_index >> 1, nz_index >> 1);
				}
			}
			buffer1 = buffer2;
			buffer2 = 0;
			tlevel--;
		}

		nx_index = nx_index >> 1;
		ny_index = ny_index >> 1;
		nz_index = nz_index >> 1;
		resolution = resolution >> 1;
		for (int index = 0; index < 27; index++)
		{
			if (((buffer1 >> index) & 1) == 1)
			{
				int nz = index / 9 - 1;
				int ny = (index - 9 * (nz + 1)) / 3 - 1;;
				int nx = index - 9 * (nz + 1) - 3 * (ny + 1) - 1;
				if ((nx_index + nx) < 0 || (nx_index + nx) >= resolution || (ny_index + ny) < 0 || (ny_index + ny) >= resolution || (nz_index + nz) < 0 || (nz_index + nz) >= resolution) continue;

				OcKey morton_now = morton3D_64_Encode_magicbits(nx_index + nx, ny_index + ny, nz_index + nz);
				MSTG_HashAddCount(((morton_now << (3 * (levelmax - tlevel) + 4)) | tlevel), nodes_buf, buf_count);
			}
		}
	}

	__global__ void DU_CountNodesBuffer(
		DArray<int> count)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= count.size()) return;

		if (count[tId] > 0)
			count[tId] = 1;
	}

	template<typename TDataType>
	void MSTsGeneratorHelper<TDataType>::DynamicUpdate(
		std::shared_ptr<AdaptiveGridSet<TDataType>> AGridSet,
		DArray<OcKey>& increaseSeed,
		DArray<OcKey>& decreaseSeed,
		Level mLevelnum,
		int mType)
	{
		GTimer timer;

		auto& nodes = AGridSet->getAGrids();
		auto& neighbors = AGridSet->getNeighbors();
		Real m_dx = AGridSet->getDx();
		Coord m_origin = AGridSet->getOrigin();
		Level m_levelmax = AGridSet->getLevelMax();
		AGridSet->setLevelNum(mLevelnum);
		assert(mLevelnum <= m_levelmax);
		AGridSet->setOctreeType(mType);

		int max_resolution = (1 << m_levelmax);
		int num_buffer = 3 * (nodes.size() - (AGridSet->getLeafNum()));
		num_buffer += (num_buffer % 2 == 0) ? 1 : 0;

		DArray<OcKey> buffer_key(num_buffer);
		buffer_key.reset();
		DArray<int> buffer_count(num_buffer);
		buffer_count.reset();

		timer.start();
		cuExecute(nodes.size(),
			DU_PutInternalNodeAndCount,
			buffer_key,
			buffer_count,
			nodes,
		    *AGridSet,
			m_levelmax,
			mType);

		cuExecute(decreaseSeed.size(),
			DU_ConstructNodesBufferDecrease,
			buffer_key,
			buffer_count,
			decreaseSeed,
			m_levelmax - mLevelnum + 1,
			m_levelmax,
			mType,
			max_resolution);

		cuExecute(increaseSeed.size(),
			DU_ConstructNodesBufferIncreaseSingle,
			buffer_key,
			buffer_count,
			increaseSeed,
			m_levelmax - mLevelnum + 1,
			m_levelmax,
			mType,
			max_resolution);

		cuExecute(num_buffer,
			DU_CountNodesBuffer,
			buffer_count);

		Reduction<int> reduce;
		int fnode_num = reduce.accumulate(buffer_count.begin(), buffer_count.size());

		Scan<int> scan;
		scan.exclusive(buffer_count.begin(), buffer_count.size());

		DArray<OcKey> nodes_morton(fnode_num);
		cuExecute(num_buffer,
			MSTG_FetchNodesNorepeat,
			nodes_morton,
			buffer_count,
			buffer_key);

		thrust::sort(thrust::device, nodes_morton.begin(), nodes_morton.begin() + nodes_morton.size(), thrust::greater<OcKey>());

		int min_resolution = (1 << (m_levelmax - mLevelnum + 1));
		int min_nodes_num = min_resolution * min_resolution * min_resolution;
		nodes.resize(min_nodes_num + 8 * fnode_num);
		cuExecute(min_nodes_num + fnode_num,
			MSTG_ComputeNodesAll,
			nodes,
			nodes_morton,
			m_levelmax - mLevelnum + 1,
			m_levelmax,
			m_origin,
			m_dx,
			min_nodes_num);

		cuExecute(fnode_num,
			MSTG_ComputeNodesChildRelationshipsSecond,
			nodes,
			nodes_morton,
			m_levelmax - mLevelnum + 1,
			m_levelmax,
			min_nodes_num);

		buffer_key.clear();
		buffer_count.clear();
		nodes_morton.clear();
	}
	DEFINE_CLASS(MSTsGeneratorHelper);
}