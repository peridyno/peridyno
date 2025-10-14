#include "PaticleUniformAnalysis.h"
#include <sstream>
#include <iostream>
#include <fstream>
#include "ParticleSystem/Module/Kernel.h"


namespace dyno {

	template<typename TDataType>
	PaticleUniformAnalysis<TDataType>::PaticleUniformAnalysis()
		: ConstraintModule()
	{
		mSummation = std::make_shared<SummationDensity<TDataType>>();
		this->inSmoothingLength()->connect(mSummation->inSmoothingLength());
		this->inSamplingDistance()->connect(mSummation->inSamplingDistance());
		this->inPosition()->connect(mSummation->inPosition());
		this->inNeighborIds()->connect(mSummation->inNeighborIds());

	};

	template<typename TDataType>
	PaticleUniformAnalysis<TDataType>::~PaticleUniformAnalysis() {

		m_SurfaceEnergy.clear();
		m_DensityEnergy.clear();
		m_TotalEnergy.clear();
		m_Count.clear();
		m_Density.clear();
		//m_output->close();

		if (m_reduce)
		{
			delete m_reduce;
		}

		if (m_arithmetic)
		{
			delete m_arithmetic;
		}

		if (m_reduce_real)
		{
			delete m_reduce_real;
		}
	};


	template <typename Real, typename Coord>
	__global__ void K_SufaceEnergy(
		DArray<Real> Energy,
		DArray<Real> density,
		DArray<Coord> posArr,
		DArrayList<int> neighbors,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		SmoothKernel<Real> kern;
		Coord pos_i = posArr[pId];

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		Coord g(0.0f);
		Real w(0.0f);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				g += kern.gradient(r, smoothingLength, 1.0) * (pos_i - posArr[j]) * (1.0f / r);
				w += kern.Weight(r, smoothingLength);
			}
		}
		if (w < EPSILON) w = EPSILON;
	
		g = g / w;

		Energy[pId] = g.dot(g);
	}



	template <typename Real>
	__global__ void K_DensityEnergy(
		DArray<Real> Energy,
		DArray<Real> density,
		Real rho_0
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Energy.size()) return;
		Real e = density[pId] < rho_0 ? rho_0 - density[pId] : 0.0f;
		Energy[pId] = e * e / (rho_0 * rho_0);

		//printf("%f\r\n", Energy[pId]);
	}


	template <typename Real>
	__global__ void K_TotalEnergy(
		DArray<Real> TotalEnergy,
		DArray<Real> DensityEnergy,
		DArray<Real> SurfaceEnergy,
		Real A_d,
		Real A_s
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= TotalEnergy.size()) return;
		
		TotalEnergy[pId] = A_d* DensityEnergy[pId] + A_s * SurfaceEnergy[pId];
		//if(TotalEnergy[pId] > 1.0)
		//printf("%f, %f + %f \r\n", TotalEnergy[pId], DensityEnergy[pId], SurfaceEnergy[pId]);
	}




	template<typename TDataType>
	void PaticleUniformAnalysis<TDataType>::constrain() {

		if (initial)
		{
			this->initializeImpl();
			initial = false;

			}

		int num = this->inPosition()->getData().size();

		if (m_TotalEnergy.size() != num)
		{
			m_DensityEnergy.resize(num);
			m_SurfaceEnergy.resize(num);
			m_TotalEnergy.resize(num);
			m_Count.resize(num);
			m_reduce = Reduction<float>::Create(num);
			m_arithmetic = Arithmetic<float>::Create(num);
			m_reduce_real = Reduction<float>::Create(num);
			m_Density.resize(num);

		}

		mSummation->varRestDensity()->setValue(1000.0f);
		mSummation->varKernelType()->setCurrentKey(0);
		mSummation->update();


		std::cout << "m_init_density: " << m_init_density << std::endl;

		cuExecute(num, K_DensityEnergy,
			m_DensityEnergy,
			mSummation->outDensity()->getData(),
			m_init_density
		)


		cuExecute(num, K_SufaceEnergy,
			m_SurfaceEnergy,
			mSummation->outDensity()->getData(),
			this->inPosition()->getData(),
			this->inNeighborIds()->getData(),
			0.0125f);

		cuExecute(num, K_TotalEnergy,
			m_TotalEnergy,
			m_DensityEnergy,
			m_SurfaceEnergy, 
			1.0f,
			0.8f
		)

		Real total_energy = m_arithmetic->Dot(m_TotalEnergy, m_TotalEnergy);
		Real total_Surface_Energy = m_arithmetic->Dot(m_SurfaceEnergy, m_SurfaceEnergy);
		
		//std::cout << "*** PaticleUniformAnalysis :" << sqrt(total_energy)/num <<  std::endl;
		std::cout << "*** total_energy: " << total_energy/num <<", Surface_Energy: "<< total_Surface_Energy/num<<  std::endl;
		if (counter % 1 == 0)
		{
			*m_output << total_energy / num << std::endl;
		}
		counter++;




	};



	template<typename TDataType>
	bool PaticleUniformAnalysis<TDataType>::initializeImpl() {
		std::cout << "PaticleUniformAnalysis initializeImpl " << std::endl;

		std::string filename = mOutpuPath + mOutputPrefix + std::string(".txt");

		m_output.reset(new std::fstream(filename.c_str(), std::ios::out));

		int num = this->inPosition()->getData().size();

		if (m_TotalEnergy.size() != num)
		{
			m_DensityEnergy.resize(num);
			m_SurfaceEnergy.resize(num);
			m_TotalEnergy.resize(num);
			m_Count.resize(num);
			m_reduce = Reduction<float>::Create(num);
			m_arithmetic = Arithmetic<float>::Create(num);
			m_reduce_real = Reduction<float>::Create(num);
			m_Density.resize(num);

		}


		mSummation->varRestDensity()->setValue(1000.0f);
		mSummation->varKernelType()->setCurrentKey(0);
		mSummation->update();

		m_init_density = m_reduce_real->maximum(
			mSummation->outDensity()->getData().begin(),
			mSummation->outDensity()->getData().size());

		return true;
	};



	template<typename TDataType>
	void PaticleUniformAnalysis<TDataType>::setNamePrefix(std::string prefix)
	{
		mOutputPrefix = prefix;
	}

	template<typename TDataType>
	void PaticleUniformAnalysis<TDataType>::setOutputPath(std::string path)
	{
		mOutpuPath = path;
	}



	DEFINE_CLASS(PaticleUniformAnalysis);
}