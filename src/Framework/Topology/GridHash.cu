#pragma once
#include "GridHash.h"
#include "Utility.h"

namespace dyno {

	__constant__ int offset[27][3] = { 0, 0, 0,
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

	template<typename TDataType>
	GridHash<TDataType>::GridHash()
	{
	}

	template<typename TDataType>
	GridHash<TDataType>::~GridHash()
	{
	}

	template<typename TDataType>
	void GridHash<TDataType>::setSpace(Real _h, Coord _lo, Coord _hi)
	{
		release();

		int padding = 2;
		ds = _h;
		lo = _lo - padding * ds;

		Coord nSeg = (_hi - _lo) / ds;

		nx = ceil(nSeg[0]) + 1 + 2 * padding;
		ny = ceil(nSeg[1]) + 1 + 2 * padding;
		nz = ceil(nSeg[2]) + 1 + 2 * padding;
		hi = lo + Coord(nx, ny, nz) * ds;

		num = nx * ny * nz;

		//		npMax = 128;

		cuSafeCall(cudaMalloc((void**)&counter, num * sizeof(int)));
		cuSafeCall(cudaMalloc((void**)&index, num * sizeof(int)));

		if (m_reduce != nullptr)
		{
			delete m_reduce;
		}

		m_reduce = Reduction<int>::Create(num);
	}

	template<typename TDataType>
	__global__ void K_CalculateParticleNumber(GridHash<TDataType> hash, Array<typename TDataType::Coord> pos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		int gId = hash.getIndex(pos[pId]);

		if (gId != INVALID)
			atomicAdd(&(hash.index[gId]), 1);
	}


	template<typename TDataType>
	__global__ void K_AddTriNumber(GridHash<TDataType> hash, Array<typename TopologyModule::Triangle> tri, Array<typename TDataType::Coord> pos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= tri.size()) return;
		
		Real ds = hash.ds;
	//	printf("%.3lf\n", ds);
		int i0 = floor((pos[tri[pId][0]][0] - hash.lo[0]) / hash.ds);
		int j0 = floor((pos[tri[pId][0]][1] - hash.lo[1]) / hash.ds);
		int k0 = floor((pos[tri[pId][0]][2] - hash.lo[2]) / hash.ds);

		int i1 = floor((pos[tri[pId][1]][0] - hash.lo[0]) / hash.ds);
		int j1 = floor((pos[tri[pId][1]][1] - hash.lo[1]) / hash.ds);
		int k1 = floor((pos[tri[pId][1]][2] - hash.lo[2]) / hash.ds);

		int i2 = floor((pos[tri[pId][2]][0] - hash.lo[0]) / hash.ds);
		int j2 = floor((pos[tri[pId][2]][1] - hash.lo[1]) / hash.ds);
		int k2 = floor((pos[tri[pId][2]][2] - hash.lo[2]) / hash.ds);

		int imin = i0 < i1 ? i0 : i1;
		imin = i2 < imin ? i2 : imin;
		int imax = i0 > i1 ? i0 : i1;
		imax = i2 > imax ? i2 : imax;

		int jmin = j0 < j1 ? j0 : j1;
		jmin = j2 < jmin ? j2 : jmin;
		int jmax = j0 > j1 ? j0 : j1;
		jmax = j2 > jmax ? j2 : jmax;

		int kmin = k0 < k1 ? k0 : k1;
		kmin = k2 < kmin ? k2 : kmin;
		int kmax = k0 > k1 ? k0 : k1;
		kmax = k2 > kmax ? k2 : kmax;

		imin--; jmin--; kmin--;
		imax++; jmax++; kmax++;

		int addi, addj, addk;
		addi = int(sqrt((Real)imax - (Real)imin + 1));
		addj = int(sqrt((Real)jmax - (Real)jmin + 1));
		addk = int(sqrt((Real)kmax - (Real)kmin + 1));

		Triangle3D t3d = Triangle3D(pos[tri[pId][0]], pos[tri[pId][1]], pos[tri[pId][2]]);
		//printf("%d %d %d\n",addi,addj,addk);
		for (int li = imin; li <= imax; li += addi)
			for (int lj = jmin; lj <= jmax; lj += addj)
				for (int lk = kmin; lk <= kmax; lk += addk)
				{
					int ri = min(imax, li + addi - 1);
					int rj = min(jmax, lj + addj - 1);
					int rk = min(kmax, lk + addk - 1);

					Real ABli = li * ds + hash.lo[0];
					Real ABri = ri * ds + ds + hash.lo[0];
					Real ABlj = lj * ds + hash.lo[1];
					Real ABrj = rj * ds + ds + hash.lo[1];
					Real ABlk = lk * ds + hash.lo[2];
					Real ABrk = rk * ds + ds + hash.lo[2];

					Coord3D ABP1 = Coord3D(ABli - 0.1 * ds * 10.0, ABlj - 0.1 * ds * 10.0, ABlk - 0.1 * ds * 10.0);
					Coord3D ABP2 = Coord3D(ABri + 0.1 * ds * 10.0, ABrj + 0.1 * ds * 10.0, ABrk + 0.1 * ds * 10.0);
					AlignedBox3D AABB = AlignedBox3D(ABP1, ABP2);

					if (AABB.meshInsert(t3d))
					{

					

						for (int i = li; i <= ri; i++)
							for (int j = lj; j <= rj; j++)
								for (int k = lk; k <= rk; k++)
								{
									Coord3D ABP11 = Coord3D(i * ds + hash.lo[0] - 0.1 * ds * 10.0,
										j * ds + hash.lo[1] - 0.1 * ds * 10.0,
										k * ds + hash.lo[2] - 0.1 * ds * 10.0);
									Coord3D ABP22 = Coord3D(i * ds + ds + hash.lo[0] + 0.1 * ds * 10.0,
										j * ds + ds + hash.lo[1] + 0.1 * ds * 10.0,
										k * ds + ds + hash.lo[2] + 0.1 * ds * 10.0);
									AlignedBox3D AABB2 = AlignedBox3D(ABP11, ABP22);

									if (AABB2.meshInsert(t3d))
									{
										int gId = hash.getIndex(i, j, k);
										if (gId != INVALID)
											atomicAdd(&(hash.index[gId]), 1);
										
									}
									
								}
					}
					
				}

	}

	template<typename TDataType>
	__global__ void K_ConstructHashTable(GridHash<TDataType> hash, Array<typename TDataType::Coord> pos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		int gId = hash.getIndex(pos[pId]);

		if (gId < 0) return;

		int index = atomicAdd(&(hash.counter[gId]), 1);
		hash.ids[hash.index[gId] + index] = pId;
	}


	template<typename TDataType>
	__global__ void K_AddTriElement(GridHash<TDataType> hash, Array<typename TopologyModule::Triangle> tri, Array<typename TDataType::Coord> pos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= tri.size()) return;
		Real ds = hash.ds;
		//Coord3D lo = hash.lo;
		//Coord3D hi = hash.hi;
		int i0 = floor((pos[tri[pId][0]][0] - hash.lo[0]) / hash.ds);
		int j0 = floor((pos[tri[pId][0]][1] - hash.lo[1]) / hash.ds);
		int k0 = floor((pos[tri[pId][0]][2] - hash.lo[2]) / hash.ds);

		int i1 = floor((pos[tri[pId][1]][0] - hash.lo[0]) / hash.ds);
		int j1 = floor((pos[tri[pId][1]][1] - hash.lo[1]) / hash.ds);
		int k1 = floor((pos[tri[pId][1]][2] - hash.lo[2]) / hash.ds);

		int i2 = floor((pos[tri[pId][2]][0] - hash.lo[0]) / hash.ds);
		int j2 = floor((pos[tri[pId][2]][1] - hash.lo[1]) / hash.ds);
		int k2 = floor((pos[tri[pId][2]][2] - hash.lo[2]) / hash.ds);

		int imin = i0 < i1 ? i0 : i1;
		imin = i2 < imin ? i2 : imin;
		int imax = i0 > i1 ? i0 : i1;
		imax = i2 > imax ? i2 : imax;

		int jmin = j0 < j1 ? j0 : j1;
		jmin = j2 < jmin ? j2 : jmin;
		int jmax = j0 > j1 ? j0 : j1;
		jmax = j2 > jmax ? j2 : jmax;

		int kmin = k0 < k1 ? k0 : k1;
		kmin = k2 < kmin ? k2 : kmin;
		int kmax = k0 > k1 ? k0 : k1;
		kmax = k2 > kmax ? k2 : kmax;

		imin--; jmin--; kmin--;
		imax++; jmax++; kmax++;

		int addi, addj, addk;
		addi = int(sqrt((Real)imax - (Real)imin + 1));
		addj = int(sqrt((Real)jmax - (Real)jmin + 1));
		addk = int(sqrt((Real)kmax - (Real)kmin + 1));

		Triangle3D t3d = Triangle3D(pos[tri[pId][0]], pos[tri[pId][1]], pos[tri[pId][2]]);


		for (int li = imin; li <= imax; li += addi)
			for (int lj = jmin; lj <= jmax; lj += addj)
				for (int lk = kmin; lk <= kmax; lk += addk)
				{
					int ri = min(imax, li + addi - 1);
					int rj = min(jmax, lj + addj - 1);
					int rk = min(kmax, lk + addk - 1);

					Real ABli = li * ds + hash.lo[0];
					Real ABri = ri * ds + ds + hash.lo[0];
					Real ABlj = lj * ds + hash.lo[1];
					Real ABrj = rj * ds + ds + hash.lo[1];
					Real ABlk = lk * ds + hash.lo[2];
					Real ABrk = rk * ds + ds + hash.lo[2];

					Coord3D ABP1 = Coord3D(ABli - 0.1 * ds * 10.0, ABlj - 0.1 * ds * 10.0, ABlk - 0.1 * ds * 10.0);
					Coord3D ABP2 = Coord3D(ABri + 0.1 * ds * 10.0, ABrj + 0.1 * ds * 10.0, ABrk + 0.1 * ds * 10.0);
					AlignedBox3D AABB = AlignedBox3D(ABP1, ABP2);


					if (AABB.meshInsert(t3d))
					{
						for (int i = li; i <= ri; i++)
							for (int j = lj; j <= rj; j++)
								for (int k = lk; k <= rk; k++)
								{
									Coord3D ABP11 = Coord3D(i * ds + hash.lo[0] - 0.1 * ds * 10.0,
										j * ds + hash.lo[1] - 0.1 * ds * 10.0,
										k * ds + hash.lo[2] - 0.1 * ds * 10.0);
									Coord3D ABP22 = Coord3D(i * ds + ds + hash.lo[0] + 0.1 * ds * 10.0,
										j * ds + ds + hash.lo[1] + 0.1 * ds * 10.0,
										k * ds + ds + hash.lo[2] + 0.1 * ds * 10.0);
									AlignedBox3D AABB2 = AlignedBox3D(ABP11, ABP22);

									if (AABB2.meshInsert(t3d))
									{
										int gId = hash.getIndex(i, j, k);

										if (gId != INVALID)
										{
											int index = atomicAdd(&(hash.counter[gId]), 1);
											hash.ids[hash.index[gId] + index] = -pId - 1;
										}

									}
								}
					}
				}
	}

	template<typename TDataType>
	void GridHash<TDataType>::construct(DeviceArray<Coord>& pos)
	{
		clear();

		dim3 pDims = int(ceil(pos.size() / BLOCK_SIZE + 0.5f));

		K_CalculateParticleNumber << <pDims, BLOCK_SIZE >> > (*this, pos);
		particle_num = m_reduce->accumulate(index, num);

		if (m_scan == nullptr)
		{
			m_scan = new Scan();
		}
		m_scan->exclusive(index, num);

		if (ids != nullptr)
		{
			cuSafeCall(cudaFree(ids));
		}
		cuSafeCall(cudaMalloc((void**)&ids, particle_num * sizeof(int)));

		//		std::cout << "Particle number: " << particle_num << std::endl;

		K_ConstructHashTable << <pDims, BLOCK_SIZE >> > (*this, pos);
		cuSynchronize();
	}


	template<typename TDataType>
	void GridHash<TDataType>::construct(DeviceArray<Coord>& pos, DeviceArray<Triangle>& tri, DeviceArray<Coord>& Tri_pos)
	{
		clear();

		dim3 pDimsTri = int(ceil(tri.size() / BLOCK_SIZE + 0.5f));

		K_AddTriNumber << <pDimsTri, BLOCK_SIZE >> > (*this, tri, Tri_pos);
		cuSynchronize();

		particle_num = m_reduce->accumulate(index, num);

		if (m_scan == nullptr)
		{
			m_scan = new Scan();
		}
		m_scan->exclusive(index, num);

		if (ids != nullptr)
		{
			cuSafeCall(cudaFree(ids));
		}
		cuSafeCall(cudaMalloc((void**)&ids, particle_num * sizeof(int)));

		
		K_AddTriElement << <pDimsTri, BLOCK_SIZE >> > (*this, tri, Tri_pos);
		
		cuSynchronize();
	}

	template<typename TDataType>
	void GridHash<TDataType>::clear()
	{
		if (counter != nullptr)
			cuSafeCall(cudaMemset(counter, 0, num * sizeof(int)));
		
		if (index != nullptr)
			cuSafeCall(cudaMemset(index, 0, num * sizeof(int)));
	}

	template<typename TDataType>
	void GridHash<TDataType>::release()
	{
		if (counter != nullptr)
			cuSafeCall(cudaFree(counter));

		if (ids != nullptr)
			cuSafeCall(cudaFree(ids));

		if (index != nullptr)
			cuSafeCall(cudaFree(index));

		// 		if (m_scan != nullptr)
		// 		{
		// 			delete m_scan;
		// 		}
	}
}