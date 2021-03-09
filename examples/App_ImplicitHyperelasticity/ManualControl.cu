#include <cuda_runtime.h>
#include "Framework/Node.h"
#include "ManualControl.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(ManualControl, TDataType)

	template<typename TDataType>
	ManualControl<TDataType>::ManualControl()
		: CustomModule()
	{

	}

	template<typename TDataType>
	ManualControl<TDataType>::~ManualControl()
	{
	}


	template<typename TDataType>
	void ManualControl<TDataType>::updateContext()
	{
		int totalNum = this->inPosition()->getValue().size();
		if (m_bFixed.size() != totalNum)
		{
			m_bFixed_host.resize(totalNum);
			m_fixed_positions_host.resize(totalNum);

			m_bFixed.resize(totalNum);
			m_fixed_positions.resize(totalNum);
		}

		for (int i = 0; i < m_bFixed_host.size(); i++)
		{
			m_bFixed_host[i] = 0;
		}

		for (auto it = m_fixedPts.begin(); it != m_fixedPts.end(); it++)
		{
			if (it->first >= 0 && it->first < totalNum)
			{
				m_bFixed_host[it->first] = 1;
				m_fixed_positions_host[it->first] = it->second;
			}
		}

		m_bFixed.assign(m_bFixed_host);
		m_fixed_positions.assign(m_fixed_positions_host);
	}

	template<typename TDataType>
	void ManualControl<TDataType>::addFixedPoint(int id, Coord pt)
	{
		m_fixedPts[id] = pt;

		bUpdateRequired = true;
	}


	template<typename TDataType>
	void ManualControl<TDataType>::removeFixedPoint(int id)
	{
		auto it = m_fixedPts.begin();
		while (it != m_fixedPts.end())
		{
			if (it->first == id)
			{
				m_fixedPts.erase(it++);
			}
			else
				it++;
		}

		bUpdateRequired = true;
	}


	template<typename TDataType>
	void ManualControl<TDataType>::clear()
	{
		m_fixedPts.clear();

		bUpdateRequired = true;
	}

	template <typename Coord>
	__global__ void K_DoFixPoints(
		GArray<Coord> curPos,
		GArray<Coord> curVel,
		GArray<Attribute> curAtts,
		GArray<int> bFixed,
		GArray<Coord> fixedPts)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= curPos.size()) return;

		if (pId == 0)
		{
			curAtts[pId].SetFixed();
		}

// 		if (bFixed[pId])
// 		{
// 			curPos[pId] = fixedPts[pId];
// 			curVel[pId] = Coord(0);
// 		}
	}

	template<typename TDataType>
	void ManualControl<TDataType>::begin()
	{
		if (bUpdateRequired)
		{
			updateContext();
			bUpdateRequired = false;
		}
	}

	template<typename TDataType>
	void ManualControl<TDataType>::applyCustomBehavior()
	{
		if (m_fixedPts.size() <= 0)
			return;

		uint pDims = cudaGridSize(m_bFixed.size(), BLOCK_SIZE);

		K_DoFixPoints<Coord> << < pDims, BLOCK_SIZE >> > (
			this->inPosition()->getValue(), 
			this->inVelocity()->getValue(),
			this->inAttribute()->getValue(),
			m_bFixed, 
			m_fixed_positions);

	}
}