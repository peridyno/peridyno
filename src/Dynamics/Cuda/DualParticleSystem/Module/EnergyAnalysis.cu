#include "EnergyAnalysis.h"
#include <sstream>
#include <iostream>
#include <fstream>


namespace dyno {

	template<typename TDataType>
	EnergyAnalysis<TDataType>::EnergyAnalysis()
		: ConstraintModule() 
	{
	};

	template<typename TDataType>
	EnergyAnalysis<TDataType>::~EnergyAnalysis() {
		
		m_Count.clear();
		m_Energy.clear();
	};

	template <typename Real, typename Coord>
	__global__ void EAM_EnergyAnalysis(
		DArray<Real> Energy,
		DArray<Coord> posArr,
		DArray<Coord> velArr,
		Real mass
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Energy[pId] = 0.5 * mass * velArr[pId].norm() * velArr[pId].norm();
	}


	template <typename Real, typename Coord>
	__global__ void EAM_NeighborCount(
		DArray<Real> count,
		DArray<Coord> pos,
		DArrayList<int> neighbors,
		Real radius
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		List<int>& list_i = neighbors[pId];
		count[pId] = (Real)(list_i.size());
	}

	template<typename TDataType>
	void EnergyAnalysis<TDataType>::constrain() {

		if (initial)
		{
			this->initializeImpl();
			initial = false;
		}

		int num = this->inPosition()->getData().size();

		if (m_Energy.size() != num)
		{
			m_Energy.resize(num);
			m_Count.resize(num);
			m_reduce = Reduction<float>::Create(num);
			m_arithmetic = Arithmetic<float>::Create(num);
			m_reduce_real = Reduction<float>::Create(num);

		}

		cuExecute(num, EAM_EnergyAnalysis,
			m_Energy,
			this->inPosition()->getData(),
			this->inVelocity()->getData(),
			1.0f);


		cuExecute(num, EAM_NeighborCount,
			m_Count,
			this->inPosition()->getData(),
			this->inNeighborIds()->getData(),
			0.0125f
		);

		Real total_energy = m_arithmetic->Dot(m_Energy, m_Energy);
		auto average_count = m_reduce_real->average(m_Count.begin(), m_Count.size());
		std::cout << "*** average_count :" << average_count << std::endl;
		if (counter % 8 == 0)
		{
			*m_output << average_count << std::endl;
		}
		counter++;

		
	};



	template<typename TDataType>
	bool EnergyAnalysis<TDataType>::initializeImpl() {
		std::cout << "EnergyAnalysis initializeImpl " << std::endl;

		std::string filename = mOutpuPath + mOutputPrefix + std::string(".txt");

		m_output.reset(new std::fstream(filename.c_str(), std::ios::out));

		return true;
	};



	template<typename TDataType>
	void EnergyAnalysis<TDataType>::setNamePrefix(std::string prefix)
	{
		mOutputPrefix = prefix;
	}

	template<typename TDataType>
	void EnergyAnalysis<TDataType>::setOutputPath(std::string path)
	{
		mOutpuPath = path;
	}



	DEFINE_CLASS(EnergyAnalysis);
}