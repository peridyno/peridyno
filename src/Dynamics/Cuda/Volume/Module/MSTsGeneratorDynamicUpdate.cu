#include "MSTsGeneratorDynamicUpdate.h"
#include "MSTsGeneratorHelper.h"
#include "Algorithm/Scan.h"
#include "Algorithm/Reduction.h"
#include <thrust/sort.h>
#include <ctime>
#include "Timer.h"
#include "STL/Stack.h"

#include <iostream>
#include <fstream>

namespace dyno
{
	IMPLEMENT_TCLASS(MSTsGeneratorDynamicUpdate, TDataType)

	template<typename TDataType>
	MSTsGeneratorDynamicUpdate<TDataType>::MSTsGeneratorDynamicUpdate()
		:AdaptiveGridGenerator<TDataType>()
	{
	}

	template<typename TDataType>
	MSTsGeneratorDynamicUpdate<TDataType>::~MSTsGeneratorDynamicUpdate()
	{
		m_seedOld.clear();
		m_seedIncrease.clear();
		m_seedDecrease.clear();
	}

	template<typename TDataType>
	void MSTsGeneratorDynamicUpdate<TDataType>::compute()
	{
		if (mDynamic == false)
		{
			auto& mseed = this->inpMorton()->getData();
			MSTsGeneratorHelper<TDataType>::ConstructionFromScratch(
				this->inAGridSet()->getDataPtr(),
				mseed,
				this->varLevelNum()->getData(),
				this->varOctreeType()->currentKey());
			mDynamic = true;
			m_seedOld.assign(mseed);
		}
		else
		{
			updateSeeds();
			MSTsGeneratorHelper<TDataType>::DynamicUpdate(
				this->inAGridSet()->getDataPtr(),
				m_seedIncrease,
				m_seedDecrease,
				this->varLevelNum()->getData(),
				this->varOctreeType()->currentKey());
			m_seedOld.assign(this->inpMorton()->getData());
		}

		AdaptiveGridGenerator<TDataType>::compute();
	}

	GPU_FUNC bool MSTDU_HashAddOld(
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
	GPU_FUNC bool MSTDU_HashAddNew(
		OcKey key,
		DArray<OcKey>& nodes,
		DArray<bool>& mark)
	{
		int index = (key * 100003) % (nodes.size());
		while (atomicCAS(&(nodes[index]), (OcKey)0, key))
		{
			if (nodes[index] == key)
			{
				mark[index] = true;
				return false;
			}
			else
				index = ((++index) % (nodes.size()));
		};
		mark[index] = true;
		return true;
	}
	GPU_FUNC bool MSTDU_HashAccess(
		OcKey key,
		DArray<OcKey>& nodes,
		DArray<bool>& mark)
	{
		int index = (key * 100003) % (nodes.size());
		while (true)
		{
			if (nodes[index] == key)
				return mark[index];
			else if (nodes[index] == 0)
				return false;
			else
				index = ((++index) % (nodes.size()));
		};
	}
	__global__ void MSTDU_OldSeedIntoHash(
		DArray<OcKey> hash_buf,
		DArray<OcKey> seeds)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= seeds.size()) return;

		MSTDU_HashAddOld(seeds[tId], hash_buf);
	}
	__global__ void MSTDU_NewSeedIntoHash(
		DArray<int> in_mark,
		DArray<OcKey> hash_buf,
		DArray<bool> hash_mark,
		DArray<OcKey> seeds)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= seeds.size()) return;

		if (MSTDU_HashAddNew(seeds[tId], hash_buf, hash_mark) == true)
			in_mark[tId] = 1;
	}
	__global__ void MSTDU_OldSeedCheck(
		DArray<int> de_mark,
		DArray<OcKey> hash_buf,
		DArray<bool> hash_mark,
		DArray<OcKey> seeds)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= seeds.size()) return;

		if (MSTDU_HashAccess(seeds[tId], hash_buf, hash_mark) == true)
			return;

		OcKey m1 = (seeds[tId] >> 3) << 3;
		for (int i = 0; i < 8; i++)
		{
			if (MSTDU_HashAccess(m1 + i, hash_buf, hash_mark) == true)
				return;
		}

		if (tId == 0)
		{
			de_mark[tId] = 1;
			return;
		}

		int j = 1;
		while (true)
		{
			if ((seeds[tId] >> 3) != (seeds[tId - j] >> 3))
			{
				de_mark[tId - j + 1] = 1;
				return;
			}
			if ((tId - j) == 0)
			{
				de_mark[tId - j] = 1;
				return;
			}
			j++;
		}
	}
	__global__ void MSTDU_CatchDynamicSeeds(
		DArray<OcKey> nodes,
		DArray<int> mark,
		DArray<OcKey> seeds)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= seeds.size()) return;

		if (tId == (seeds.size() - 1) && mark[tId] < nodes.size())
		{
			nodes[mark[tId]] = seeds[tId];
			return;
		}
		else if (tId < (seeds.size() - 1) && mark[tId] != mark[tId + 1])
		{
			nodes[mark[tId]] = seeds[tId];
			return;
		}
	}
	template<typename TDataType>
	void MSTsGeneratorDynamicUpdate<TDataType>::updateSeeds()
	{
		auto& seed_new = this->inpMorton()->getData();

		int buf_num = 3 * m_seedOld.size();
		DArray<OcKey> hash_buffer(buf_num);
		hash_buffer.reset();
		DArray<bool> hash_mark(buf_num);
		hash_mark.reset();

		cuExecute(m_seedOld.size(),
			MSTDU_OldSeedIntoHash,
			hash_buffer,
			m_seedOld);

		DArray<int> increase_mark(seed_new.size());
		increase_mark.reset();
		cuExecute(seed_new.size(),
			MSTDU_NewSeedIntoHash,
			increase_mark,
			hash_buffer,
			hash_mark,
			seed_new);

		DArray<int> decrease_mark(m_seedOld.size());
		decrease_mark.reset();
		cuExecute(m_seedOld.size(),
			MSTDU_OldSeedCheck,
			decrease_mark,
			hash_buffer,
			hash_mark,
			m_seedOld);

		Reduction<int> reduce;
		int increase_num = reduce.accumulate(increase_mark.begin(), increase_mark.size());
		Scan<int> scan;
		scan.exclusive(increase_mark.begin(), increase_mark.size());

		int decrease_num = reduce.accumulate(decrease_mark.begin(), decrease_mark.size());
		scan.exclusive(decrease_mark.begin(), decrease_mark.size());
		printf("Dynamic update!!! %d %d \n", increase_num, decrease_num);

		m_seedIncrease.resize(increase_num);
		cuExecute(seed_new.size(),
			MSTDU_CatchDynamicSeeds,
			m_seedIncrease,
			increase_mark,
			seed_new);
		thrust::sort(thrust::device, m_seedIncrease.begin(), m_seedIncrease.begin() + m_seedIncrease.size());

		m_seedDecrease.resize(decrease_num);
		cuExecute(m_seedOld.size(),
			MSTDU_CatchDynamicSeeds,
			m_seedDecrease,
			decrease_mark,
			m_seedOld);

		hash_buffer.clear();
		hash_mark.clear();
		increase_mark.clear();
		decrease_mark.clear();
	}

	DEFINE_CLASS(MSTsGeneratorDynamicUpdate);
}