#include <cuda_runtime.h>
#include "Log.h"
#include "Node.h"
#include "FixedPoints.h"

namespace dyno
{
	IMPLEMENT_TCLASS(FixedPoints, TDataType)

	template<typename TDataType>
	FixedPoints<TDataType>::FixedPoints()
		: ConstraintModule()
	{
	}

	template<typename TDataType>
	FixedPoints<TDataType>::~FixedPoints()
	{
		m_bFixed.clear();
		m_fixed_positions.clear();
	}


	template<typename TDataType>
	bool FixedPoints<TDataType>::initializeImpl()
	{
		if (m_position.isEmpty() || m_velocity.isEmpty())
		{
			std::cout << "Exception: " << std::string("FixedPoints's fields are not fully initialized!") << "\n";
			return false;
		}

		return true;
	}


	template<typename TDataType>
	void FixedPoints<TDataType>::updateContext()
	{
		int totalNum = m_position.getData().size();
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
	void FixedPoints<TDataType>::addFixedPoint(int id, Coord pt)
	{
		m_fixedPts[id] = pt;

		bUpdateRequired = true;
	}


	template<typename TDataType>
	void FixedPoints<TDataType>::removeFixedPoint(int id)
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
	void FixedPoints<TDataType>::clear()
	{
		m_fixedPts.clear();

		bUpdateRequired = true;
	}

	template <typename Coord>
	__global__ void K_DoFixPoints(
		DArray<Coord> curPos,
		DArray<Coord> curVel,
		DArray<int> bFixed,
		DArray<Coord> fixedPts)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= curPos.size()) return;
		
		if (bFixed[pId])
		{
			curPos[pId] = fixedPts[pId];
			curVel[pId] = Coord(0);
		}

	}

	template<typename TDataType>
	void FixedPoints<TDataType>::constrain()
	{
		if (m_fixedPts.size() <= 0)
			return;

		if (bUpdateRequired)
		{
			updateContext();
			bUpdateRequired = false;
		}


		uint pDims = cudaGridSize(m_bFixed.size(), BLOCK_SIZE);

		K_DoFixPoints<Coord> << < pDims, BLOCK_SIZE >> > (m_position.getData(), m_velocity.getData(), m_bFixed, m_fixed_positions);
	}

	template <typename Coord>
	__global__ void K_DoPlaneConstrain(
		DArray<Coord> curPos,
		Coord origin,
		Coord dir)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= curPos.size()) return;

		float tmp = dir.dot(curPos[pId] - origin);
		if (tmp < 0)
		{
			curPos[pId] -= tmp*dir;
		}
	}

	template<typename TDataType>
	void FixedPoints<TDataType>::constrainPositionToPlane(Coord pos, Coord dir)
	{
		uint pDims = cudaGridSize(m_bFixed.size(), BLOCK_SIZE);

		K_DoPlaneConstrain<< < pDims, BLOCK_SIZE >> > (m_position.getData(), pos, dir);
	}

	DEFINE_CLASS(FixedPoints);
}