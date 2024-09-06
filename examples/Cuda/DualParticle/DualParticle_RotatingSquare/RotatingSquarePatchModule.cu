#include <cuda_runtime.h>
#include "RotatingSquarePatchModule.h"
#include "Node.h"
#include "SceneGraph.h"

namespace dyno
{
	

	template<typename TDataType>
	RotatingSquarePatchModule<TDataType>::RotatingSquarePatchModule()
		: NumericalIntegrator()
	{
		this->inAttribute()->tagOptional(true);
	}

	template<typename TDataType>
	void RotatingSquarePatchModule<TDataType>::begin()
	{
		if (!this->inPosition()->isEmpty())
		{
			int num = this->inPosition()->size();
			m_prePosition.resize(num);
			m_preVelocity.resize(num);
			m_prePosition.assign(this->inPosition()->getData());
			m_preVelocity.assign(this->inVelocity()->getData());
			this->inForceDensity()->getDataPtr()->reset();
		}
	}

	template<typename TDataType>
	void RotatingSquarePatchModule<TDataType>::end()
	{

	}

	template<typename Real, typename Coord>
	__global__ void KRS_UpdateVelocity(
		DArray<Coord> vel,
		DArray<Coord> forceDensity,
		Coord gravity,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= forceDensity.size()) return;
		vel[pId] += dt * (forceDensity[pId]);
		vel[pId][1] = 0.0f;
	}




	template<typename Real, typename Coord>
	__global__ void KRS_InitVelocity(
		DArray<Coord> vel,
		DArray<Coord> forceDensity,
		DArray<Coord> pos,
		Coord Origin,
		Real T,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= forceDensity.size()) return;
			
			Origin[1] = pos[pId][1];

			Coord x = pos[pId] - Origin;
			Real r = x.norm();

			Real f = 1.0f  / T;
			Real phi = 2 * M_PI * f;
			Real v = phi * r;
		
			if (r < EPSILON) r = EPSILON;

			vel[pId][0] = -v * x[2] / r;
			vel[pId][2] = v * x[0] / r;
	}


	template<typename TDataType>
	bool RotatingSquarePatchModule<TDataType>::updateVelocity()
	{
		Real dt = Real(0.001);
		if(this->getParentNode() != NULL)
			dt = getParentNode()->getDt();
		Coord gravity = SceneGraph::getInstance().getGravity();

		int total_num = this->inPosition()->size();

		if (this->inFrameNumber()->getValue() == 0)
		{
			auto& points = this->inPosition()->getData();

			Reduction<Coord> reduce;
			Coord hiBound = reduce.maximum(points.begin(), points.size());
			Coord loBound = reduce.minimum(points.begin(), points.size());
			Real averageHeight = reduce.average(points.begin(), points.size())[1];

			Coord origin = (hiBound - loBound) / 2.0f + loBound;
			origin[1] = averageHeight;

			std::cout << "Rotating: " << origin << std::endl;
			Real T = 2 * M_PI / this->varInitialAngularVelocity()->getValue();

			cuExecute(total_num,
				KRS_InitVelocity,
				this->inVelocity()->getData(),
				this->inForceDensity()->getData(),
				this->inPosition()->getData(),
				origin,
				T,
				dt);
		}


		cuExecute(total_num,
			KRS_UpdateVelocity,
			this->inVelocity()->getData(),
			this->inForceDensity()->getData(),
			gravity,
			dt);

		return true;
	}

	template<typename Real, typename Coord>
	__global__ void KRS_UpdatePosition(
		DArray<Coord> pos,
		DArray<Coord> vel,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		pos[pId] += dt * vel[pId];
	}




	template<typename TDataType>
	bool RotatingSquarePatchModule<TDataType>::updatePosition()
	{
		cudaDeviceSynchronize();
		//TODO: 
		Real dt = 0.001;
		dt = this->inTimeStep()->getData();
		std::cout <<"*ROTATING 2D INTEGRATOR:: Time step : " << this->inTimeStep()->getData() << std::endl;

		int total_num = this->inPosition()->getDataPtr()->size();

		cuExecute(total_num,
			KRS_UpdatePosition,
			this->inPosition()->getData(),
			this->inVelocity()->getData(),
			dt);


		return true;
	}

	template<typename TDataType>
	bool RotatingSquarePatchModule<TDataType>::integrate()
	{
		if (!this->inPosition()->isEmpty())
		{
			updateVelocity();
			updatePosition();
		}

		return true;
	}


	template<typename TDataType>
	void RotatingSquarePatchModule<TDataType>::updateImpl()
	{
		this->begin();
		this->integrate();
		this->end();
		fragNum++;
	}



	DEFINE_CLASS(RotatingSquarePatchModule);
}