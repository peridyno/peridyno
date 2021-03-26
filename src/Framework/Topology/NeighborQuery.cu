#include <cuda_runtime.h>
#include "NeighborQuery.h"
#include "Framework/Node.h"
#include "Topology/NeighborList.h"
#include "Topology/FieldNeighbor.h"
#include "Framework/SceneGraph.h"
#include "Algorithm/Scan.h"

namespace dyno
{
	__constant__ int offset1[27][3] = { 0, 0, 0,
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

	IMPLEMENT_CLASS_1(NeighborQuery, TDataType)

	template<typename TDataType>
	NeighborQuery<TDataType>::NeighborQuery()
		: ComputeModule()
		, m_maxNum(0)
	{

		Vector3f sceneLow = SceneGraph::getInstance().getLowerBound();
		Vector3f sceneUp = SceneGraph::getInstance().getUpperBound();

		m_lowBound = Coord(sceneLow[0], sceneLow[1], sceneLow[2]);
		m_highBound = Coord(sceneUp[0], sceneUp[1], sceneUp[2]);
		this->inRadius()->setValue(Real(0.011));

		m_hash.setSpace(this->inRadius()->getValue(), m_lowBound, m_highBound);
	}


	template<typename TDataType>
	NeighborQuery<TDataType>::NeighborQuery(DArray<Coord>& position)
		: ComputeModule()
	{
		Vector3f sceneLow = SceneGraph::getInstance().getLowerBound();
		Vector3f sceneUp = SceneGraph::getInstance().getUpperBound();

		m_lowBound = Coord(sceneLow[0], sceneLow[1], sceneLow[2]);
		m_highBound = Coord(sceneUp[0], sceneUp[1], sceneUp[2]);
		this->inRadius()->setValue(Real(0.011));

		this->inPosition()->setElementCount(position.size());
		this->inPosition()->getValue().assign(position);
	}

	template<typename TDataType>
	NeighborQuery<TDataType>::~NeighborQuery()
	{
		m_hash.release();
	}

	template<typename TDataType>
	NeighborQuery<TDataType>::NeighborQuery(Real s, Coord lo, Coord hi)
		: ComputeModule()
		, m_maxNum(0)
	{
		this->inRadius()->setValue(Real(s));

		m_lowBound = lo;
		m_highBound = hi;
	}

	template<typename TDataType>
	bool NeighborQuery<TDataType>::initializeImpl()
	{
		if (!this->inPosition()->isEmpty() && this->outNeighborhood()->isEmpty())
		{
			this->outNeighborhood()->setElementCount(this->inPosition()->getElementCount(), m_maxNum);
		}

		if (this->inPosition()->isEmpty() || this->inRadius()->isEmpty())
		{
			std::cout << "Exception: " << std::string("NeighborQuery's fields are not fully initialized!") << "\n";
			return false;
		}

		int pNum = this->inPosition()->getElementCount();

		CArray<Coord> hostPos;
		hostPos.resize(pNum);

// 		Function1Pt::copy(hostPos, m_position.getValue());
// 
// 		m_lowBound = Vector3f(10000000, 10000000, 10000000);
// 		m_highBound = Vector3f(-10000000, -10000000, -10000000);
// 
// 		for (int i = 0; i < pNum; i++)
// 		{
// 			m_lowBound[0] = min(hostPos[i][0], m_lowBound[0]);
// 			m_lowBound[1] = min(hostPos[i][1], m_lowBound[1]);
// 			m_lowBound[2] = min(hostPos[i][2], m_lowBound[2]);
// 
// 			m_highBound[0] = max(hostPos[i][0], m_highBound[0]);
// 			m_highBound[1] = max(hostPos[i][1], m_highBound[1]);
// 			m_highBound[2] = max(hostPos[i][2], m_highBound[2]);
// 		}

	
		m_hash.setSpace(this->inRadius()->getValue(), m_lowBound, m_highBound);

//		m_reduce = Reduction<int>::Create(m_position.getElementCount());
		triangle_first = true;

		printf("FROM nbrPoint: %d\n", pNum);
		compute();

		return true;
	}

	template<typename TDataType>
	void NeighborQuery<TDataType>::compute()
	{
		if (!this->inPosition()->isEmpty())
		{
			int p_num = this->inPosition()->getElementCount();
			if (p_num <= 0)
				return;

			if (this->outNeighborhood()->getElementCount() != p_num)
			{
				this->outNeighborhood()->setElementCount(p_num);
			}

			m_hash.clear();
			m_hash.construct(this->inPosition()->getValue());

			if (!this->outNeighborhood()->getValue().isLimited())
			{
				queryNeighborDynamic(this->outNeighborhood()->getValue(), this->inPosition()->getValue(), this->inRadius()->getValue());
			}
			else
			{
				queryNeighborFixed(this->outNeighborhood()->getValue(), this->inPosition()->getValue(), this->inRadius()->getValue());
			}
		}
	}


	template<typename TDataType>
	void NeighborQuery<TDataType>::setBoundingBox(Coord lowerBound, Coord upperBound)
	{
		m_lowBound = lowerBound;
		m_highBound = upperBound;
	}

	template<typename TDataType>
	void NeighborQuery<TDataType>::queryParticleNeighbors(NeighborList<int>& nbr, DArray<Coord>& pos, Real radius)
	{
		CArray<Coord> hostPos;
		hostPos.resize(pos.size());

		hostPos.assign(pos);

// 		m_lowBound = Vector3f(10000000, 10000000, 10000000);
// 		m_highBound = Vector3f(-10000000, -10000000, -10000000);
// 
// 		for (int i = 0; i < pos.size(); i++)
// 		{
// 			m_lowBound[0] = min(hostPos[i][0], m_lowBound[0]);
// 			m_lowBound[1] = min(hostPos[i][1], m_lowBound[1]);
// 			m_lowBound[2] = min(hostPos[i][2], m_lowBound[2]);
// 
// 			m_highBound[0] = max(hostPos[i][0], m_highBound[0]);
// 			m_highBound[1] = max(hostPos[i][1], m_highBound[1]);
// 			m_highBound[2] = max(hostPos[i][2], m_highBound[2]);
// 		}

		m_hash.setSpace(radius, m_lowBound, m_highBound);
		m_hash.construct(this->inPosition()->getValue());

		if (!nbr.isLimited())
		{
			queryNeighborDynamic(nbr, pos, radius);
		}
		else
		{
			queryNeighborFixed(nbr, pos, radius);
		}
	}

	template<typename Real, typename Coord, typename TDataType>
	__global__ void K_CalNeighborSize(
		DArray<int> count,
		DArray<Coord> position_new,
		DArray<Coord> position, 
		GridHash<TDataType> hash, 
		Real h)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position_new.size()) return;

		Coord pos_ijk = position_new[pId];
		int3 gId3 = hash.getIndex3(pos_ijk);

		int counter = 0;
		for (int c = 0; c < 27; c++)
		{
			int cId = hash.getIndex(gId3.x + offset1[c][0], gId3.y + offset1[c][1], gId3.z + offset1[c][2]);
			if (cId >= 0) {
				int totalNum = hash.getCounter(cId);
				for (int i = 0; i < totalNum; i++) {
					int nbId = hash.getParticleId(cId, i);
					Real d_ij = (pos_ijk - position[nbId]).norm();
					if (d_ij < h)
					{
						counter++;
					}
				}
			}
		}

		count[pId] = counter;
	}
	

	template<typename Real, typename Coord, typename TDataType>
	__global__ void K_GetNeighborElements(
		NeighborList<int> nbr,
		DArray<Coord> position_new,
		DArray<Coord> position, 
		GridHash<TDataType> hash, 
		Real h)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position_new.size()) return;

		Coord pos_ijk = position_new[pId];
		int3 gId3 = hash.getIndex3(pos_ijk);

		int j = 0;
		for (int c = 0; c < 27; c++)
		{
			int cId = hash.getIndex(gId3.x + offset1[c][0], gId3.y + offset1[c][1], gId3.z + offset1[c][2]);
			if (cId >= 0) {
				int totalNum = hash.getCounter(cId);// min(hash.getCounter(cId), hash.npMax);
				for (int i = 0; i < totalNum; i++) {
					int nbId = hash.getParticleId(cId, i);
					Real d_ij = (pos_ijk - position[nbId]).norm();
					if (d_ij < h)
					{
						nbr.setElement(pId, j, nbId);
						j++;
					}
				}
			}
		}
	}

	template<typename TDataType>
	void NeighborQuery<TDataType>::queryNeighborSize(DArray<int>& num, DArray<Coord>& pos, Real h)
	{
		uint pDims = cudaGridSize(num.size(), BLOCK_SIZE);
		K_CalNeighborSize << <pDims, BLOCK_SIZE >> > (num, pos, this->inPosition()->getValue(), m_hash, h);
		cuSynchronize();
	}

	template<typename TDataType>
	void NeighborQuery<TDataType>::queryNeighborDynamic(NeighborList<int>& nbrList, DArray<Coord>& pos, Real h)
	{
		if (pos.size() <= 0)
		{
			return;
		}

		DArray<int>& nbrNum = nbrList.getIndex();
		if (nbrNum.size() != pos.size())
			nbrList.resize(pos.size());

		queryNeighborSize(nbrNum, pos, h);

		int sum = m_reduce.accumulate(nbrNum.begin(), nbrNum.size());

		m_scan.exclusive(nbrNum, true);
		cuSynchronize();


		if (sum > 0)
		{
			DArray<int>& elements = nbrList.getElements();
			elements.resize(sum);

			uint pDims = cudaGridSize(pos.size(), BLOCK_SIZE);
			K_GetNeighborElements << <pDims, BLOCK_SIZE >> > (nbrList, pos, this->inPosition()->getValue(), m_hash, h);
			cuSynchronize();
		}
	}
	
	template<typename Real, typename Coord, typename TDataType>
	__global__ void K_ComputeNeighborFixed(
		NeighborList<int> neighbors, 
		DArray<Coord> position_new,
		DArray<Coord> position, 
		GridHash<TDataType> hash, 
		Real h,
		int* heapIDs,
		Real* heapDistance)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position_new.size()) return;

		int nbrLimit = neighbors.getNeighborLimit();

		int* ids(heapIDs + pId * nbrLimit);// = new int[nbrLimit];
		Real* distance(heapDistance + pId * nbrLimit);// = new Real[nbrLimit];

		Coord pos_ijk = position_new[pId];
		int3 gId3 = hash.getIndex3(pos_ijk);

		int counter = 0;
		for (int c = 0; c < 27; c++)
		{
			int cId = hash.getIndex(gId3.x + offset1[c][0], gId3.y + offset1[c][1], gId3.z + offset1[c][2]);
			if (cId >= 0) {
				int totalNum = hash.getCounter(cId);// min(hash.getCounter(cId), hash.npMax);
				for (int i = 0; i < totalNum; i++) {
					int nbId = hash.getParticleId(cId, i);
					float d_ij = (pos_ijk - position[nbId]).norm();
					if (d_ij < h)
					{
						if (counter < nbrLimit)
						{
							ids[counter] = nbId;
							distance[counter] = d_ij;
							counter++;
						}
						else
						{
							int maxId = 0;
							float maxDist = distance[0];
							for (int ne = 1; ne < nbrLimit; ne++)
							{
								if (maxDist < distance[ne])
								{
									maxDist = distance[ne];
									maxId = ne;
								}
							}
							if (d_ij < distance[maxId])
							{
								distance[maxId] = d_ij;
								ids[maxId] = nbId;
							}
						}
					}
				}
			}
		}

		neighbors.setNeighborSize(pId, counter);

		int bId;
		for (bId = 0; bId < counter; bId++)
		{
			neighbors.setElement(pId, bId, ids[bId]);
		}
	}

	template<typename TDataType>
	void NeighborQuery<TDataType>::queryNeighborFixed(NeighborList<int>& nbrList, DArray<Coord>& pos, Real h)
	{
		int num = pos.size();
		int* ids;
		Real* distance;
		cuSafeCall(cudaMalloc((void**)&ids, num * sizeof(int) * nbrList.getNeighborLimit()));
		cuSafeCall(cudaMalloc((void**)&distance, num * sizeof(int) * nbrList.getNeighborLimit()));

		uint pDims = cudaGridSize(num, BLOCK_SIZE);
		K_ComputeNeighborFixed << <pDims, BLOCK_SIZE >> > (
			nbrList, 
			pos, 
			this->inPosition()->getValue(), 
			m_hash, 
			h, 
			ids, 
			distance);
		cuSynchronize();

		cuSafeCall(cudaFree(ids));
		cuSafeCall(cudaFree(distance));
	}
}