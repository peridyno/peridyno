#include "DevicePoissonDiskSampler.h"
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <chrono>

namespace dyno
{

	template<typename Real, typename Coord>
	__global__ void DPDS_RandomPos(
		DArray<Coord> positions,
		Coord lower,
		Coord upper,
		Real dx,
		int offset
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= positions.size()) return;

		unsigned long seed = (Real)(pId) + offset;
 
		curandState state;
		curand_init(seed, pId, 0, &state);

		positions[pId][0] = lower[0] + curand_uniform(&state) * (upper[0] - lower[0]);
		positions[pId][1] = lower[1] + curand_uniform(&state) * (upper[1] - lower[1]);
		positions[pId][2] = lower[2] + curand_uniform(&state) * (upper[2] - lower[2]);

	}

	template<typename Real, typename Coord>
	__global__ void DPDS_UpdatePosition(
		DArray<Coord> pos,
		DArray<Coord> vel,
		Real delta)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;
		pos[pId] += delta * (vel[pId]);
	}



	//template<typename TDataType>
	//void DevicePoissonDiskSampler<TDataType>::updatePositions()
	//{

	//	int num = this->statePosition()->size();

	//	cuExecute(num,
	//		DPDS_UpdatePosition,
	//		this->statePosition()->getData(),
	//		this->stateVirtualVelocity()->getData(),
	//		this->varDelta()->getValue()
	//	);
	//}


	template <typename TDataType>
	DevicePoissonDiskSampler<TDataType>::DevicePoissonDiskSampler()
		: SdfSampler<TDataType>()
	{
		this->varSpacing()->setRange(0.004, 1.0);
		this->stateNeighborLength()->setValue(this->varSpacing()->getValue() * 1.1f);
		this->statePosition()->allocate();
		//this->stateVirtualVelocity()->allocate();

		m_neighbor = std::make_shared<NeighborPointQuery<TDataType>>();
		this->stateNeighborLength()->connect(m_neighbor->inRadius());
		this->statePosition()->connect(m_neighbor->inPosition());

		m_constraint = std::make_shared<PoissionDiskPositionShifting<TDataType>>();
		m_constraint->varIterationNumber()->setValue(50);
		m_constraint->varKernelType()->getDataPtr()->setCurrentKey(1);
		m_constraint->varRestDensity()->setValue(1000.0);
		this->varSpacing()->connect(m_constraint->inSamplingDistance());
		this->stateNeighborLength()->connect(m_constraint->inSmoothingLength());
		this->varDelta()->connect(m_constraint->inDelta());
		this->statePosition()->connect(m_constraint->inPosition());
		//this->stateVirtualVelocity()->connect(m_constraint->inVelocity());
		m_neighbor->outNeighborIds()->connect(m_constraint->inNeighborIds());

		//ptr_viscosity = std::make_shared<ImplicitViscosity<TDataType>>();
		//ptr_viscosity->varViscosity()->setValue(Real(1050.0));
		//this->varDelta()->connect(ptr_viscosity->inTimeStep());
		//this->stateNeighborLength()->connect(ptr_viscosity->inSmoothingLength());
		//this->statePosition()->connect(ptr_viscosity->inPosition());
		//this->stateVirtualVelocity()->connect(ptr_viscosity->inVelocity());
		//m_neighbor->outNeighborIds()->connect(ptr_viscosity->inNeighborIds());

	};


	template <typename TDataType>
	DevicePoissonDiskSampler<TDataType>::~DevicePoissonDiskSampler()
	{
		mMinimumDistances.clear();
		mPointsInsideSdf.clear();
		mInsideSdfCounters.clear();
	};

	template<typename Real, typename Coord>
	__global__ void DPDS_MinDistance(
		DArray<Real> minDis, 
		DArray<Coord> positions,
		DArrayList<int> neighbors,
		Real neighborLength
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= positions.size()) return;


		Real minimumR(10000.0f);
	
		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];

			if (j == pId) continue;

			Real r = (positions[pId] - positions[j]).norm();
			if (minimumR > r)
			{
				minimumR = r;
			}
		}
		if(minimumR > neighborLength) 
			minDis[pId] = neighborLength;
		else
			minDis[pId] = minimumR;

		//if (pId < 10) printf("%d -- %f \r\n",pId, minimumR);
	}



	template<typename TDataType, typename Real, typename Coord>
	__global__ void DPDS_PonitInsideSDF(
		DArray<int> counter,
		DistanceField3D<TDataType> inputSDF,
		DArray<Coord> points,
		Real dx
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= points.size()) return;

		Real a;
		Coord normal;
		inputSDF.getDistance(points[pId], a, normal);

		if (a < 0.0f)
		{
			counter[pId] = 1;
		}
		else
		{
			counter[pId] = 0;
		}

	}

	template<typename Coord>
	__global__ void DPDS_GetLastPoints
	(
		DArray<Coord>  LastPoints,
		DArray<Coord>  post_elements,
		DArray<int> counter
	)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= post_elements.size()) return;

		if (id == post_elements.size() - 1 || counter[id] != counter[id + 1])
		{
			LastPoints[counter[id]] = post_elements[id];
		}
	}

	template <typename TDataType>
	TDataType::Real DevicePoissonDiskSampler<TDataType>::minimumDistanceEstimation()
	{
		Real error(0.0f);

		int num = this->statePosition()->size();

		cuExecute(num,
			DPDS_MinDistance,
			mMinimumDistances,
			this->statePosition()->getData(),
			m_neighbor->outNeighborIds()->getData(),
			this->stateNeighborLength()->getValue()
		);

		Real average_minimum = mReduceReal.average(mMinimumDistances.begin(), mMinimumDistances.size());
		Real min_minimum = mReduceReal.minimum(mMinimumDistances.begin(), mMinimumDistances.size());

		//std::cout << "AVR " << average_minimum <<", MIN " << min_minimum << std::endl;

		return error = min_minimum;

	}

	template <typename TDataType>
	void DevicePoissonDiskSampler<TDataType>::imposeConstraint()
	{
	
		Real minDistance = 100.0f;
		Real old_minDistance = 0.0f;

		int iter = 0;
		while(
			((abs((minDistance - old_minDistance) / this->varSpacing()->getValue()) > 0.0001f)
			)
			&&
			(iter < this->varMaxIteration()->getValue())
			)
		{
			iter++;

			std::cout << "." ;

			m_neighbor->update();

			old_minDistance = minDistance;
			minDistance = this->minimumDistanceEstimation();

			m_constraint->varIterationNumber()->setValue(50);
			m_constraint->update();

			//ptr_viscosity->update();
			//this->updatePositions();

			if (iter == this->varMaxIteration()->getValue())
			{
				std::cout << std::endl;
				std::cout << "Minimum Distance of points:" << minDistance << std::endl;
			}
		}

	}

	template <typename TDataType>
	void  DevicePoissonDiskSampler<TDataType>::resizeArrays(int num)
	{
		mMinimumDistances.resize(num);
		mInsideSdfCounters.resize(num);
	}


	template <typename TDataType>
	void DevicePoissonDiskSampler<TDataType>::resetStates()
	{
		this->stateNeighborLength()->setValue(this->varSpacing()->getValue() * 1.1f);

		Real dx = this->varSpacing()->getData();


		if (this->getVolumeOctree() != nullptr)
		{
			m_inputSDF = this->convert2Uniform(this->getVolumeOctree(), dx);
		}
		else if (this->getVolume() != nullptr)
		{
			m_inputSDF = std::make_shared<dyno::DistanceField3D<TDataType>>();
			m_inputSDF->assign(this->getVolume()->stateLevelSet()->getData().getSDF());
		}
		else
		{
			return;
		}

		if (this->statePointSet()->isEmpty()) {
			auto pts = std::make_shared<PointSet<TDataType>>();
			this->statePointSet()->setDataPtr(pts);
		}

		Coord lower = m_inputSDF->lowerBound();
		Coord upper = m_inputSDF->upperBound();

		unsigned int candidate_number = 0.9 * (upper[0] - lower[0]) * (upper[1] - lower[1]) * (upper[2] - lower[2]) / pow(dx, 3);

		this->statePosition()->resize(candidate_number);
		//this->stateVirtualVelocity()->resize(candidate_number);

		this->resizeArrays(candidate_number);

		//std::cout << lower << " - " << upper <<  " total number " << candidate_number << std::endl;
		//std::cout <<"DX " << this->varSpacing()->getValue() << ", H " << this->stateNeighborLength()->getValue() << std::endl;

		auto now = std::chrono::system_clock::now();
		m_seed_offset += abs(static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count())) % 100000;
		std::cout << m_seed_offset << std::endl;

		cuExecute(candidate_number,
			DPDS_RandomPos,
			this->statePosition()->getData(),
			lower,
			upper,
			dx,
			m_seed_offset
		);

		if (!this->varConstraintDisable()->getValue())
		{
			std::cout << "GPU Poisson-disk Points";
			this->imposeConstraint();
		}

		cuExecute(candidate_number,
			DPDS_PonitInsideSDF,
			mInsideSdfCounters,
			*m_inputSDF,
			this->statePosition()->getData(),
			dx
		);

		int totalNum = mReduceInt.accumulate(mInsideSdfCounters.begin(), mInsideSdfCounters.size());
		mScan.exclusive(mInsideSdfCounters.begin(), mInsideSdfCounters.size());

		mPointsInsideSdf.resize(totalNum);

		cuExecute(this->statePosition()->size(),
			DPDS_GetLastPoints,
			mPointsInsideSdf,
			this->statePosition()->getData(),
			mInsideSdfCounters
		);

		this->statePosition()->assign(mPointsInsideSdf);

		if (this->statePosition()->size() >= 0) {
			auto topo = this->statePointSet()->getDataPtr();
			topo->setPoints(this->statePosition()->getData());
			topo->update();
		}

	};

	DEFINE_CLASS(DevicePoissonDiskSampler);


}