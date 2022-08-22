#include <cuda_runtime.h>
#include "ParticleIntegrator.h"
#include "Node.h"
#include "SceneGraphFactory.h"

namespace dyno
{
	//IMPLEMENT_TCLASS(ParticleIntegrator, TDataType)

	template<typename TDataType>
	ParticleIntegrator<TDataType>::ParticleIntegrator()
		: NumericalIntegrator()
	{
		this->inAttribute()->tagOptional(true);
	}

	template<typename TDataType>
	void ParticleIntegrator<TDataType>::begin()
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
	void ParticleIntegrator<TDataType>::end()
	{

	}

	template<typename Real, typename Coord>
	__global__ void K_UpdateVelocity(
		DArray<Coord> vel,
		DArray<Coord> forceDensity,
		Coord gravity,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= forceDensity.size()) return;

		vel[pId] += dt * (forceDensity[pId] + gravity);
	}


	template<typename Real, typename Coord>
	__global__ void K_UpdateVelocity(
		DArray<Coord> vel,
		DArray<Coord> forceDensity,
		DArray<Coord> positionLast,
		DArray<Coord> positionNow,
		Coord gravity,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= forceDensity.size()) return;

		//if (positionNow[pId][1] < 0 && positionLast[pId][1] >= 0)
		//	dt *= positionLast[pId][1] / (positionLast[pId][1] - positionNow[pId][1]);

		vel[pId] += dt * (forceDensity[pId] + gravity);
	}


	template<typename Real, typename Coord>
	__global__ void K_UpdateVelocity(
		DArray<Coord> vel,
		DArray<Coord> forceDensity,
		DArray<Attribute> atts,
		Coord gravity,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= forceDensity.size()) return;

		Attribute att = atts[pId];

		if (att.isDynamic())
		{
			vel[pId] += dt * (forceDensity[pId] + gravity);
		}
	}


	template<typename Real, typename Coord>
	__global__ void K_UpdateVelocity(
		DArray<Coord> vel,
		DArray<Coord> forceDensity,
		DArray<Coord> positionLast,
		DArray<Coord> positionNow,
		DArray<Attribute> atts,
		Coord gravity,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= forceDensity.size()) return;

		Attribute att = atts[pId];

		if (att.isDynamic())
		{
			vel[pId] += dt * (forceDensity[pId] + gravity);
		}
	}

	template<typename TDataType>
	bool ParticleIntegrator<TDataType>::updateVelocity()
	{
		Real dt = this->inTimeStep()->getData();

		auto scn = dyno::SceneGraphFactory::instance()->active();
		Coord gravity = scn->getGravity();

		int total_num = this->inPosition()->size();

		if (this->inAttribute()->isEmpty())
		{
			cuExecute(total_num,
				K_UpdateVelocity,
				this->inVelocity()->getData(),
				this->inForceDensity()->getData(),
				m_prePosition,
				this->inPosition()->getData(),
				gravity,
				dt);
		}
		else
		{
			cuExecute(total_num,
				K_UpdateVelocity,
				this->inVelocity()->getData(),
				this->inForceDensity()->getData(),
				m_prePosition,
				this->inPosition()->getData(),
				this->inAttribute()->getData(),
				gravity,
				dt);
		}
		

		return true;
	}

	template<typename Real, typename Coord>
	__global__ void K_UpdatePosition(
		DArray<Coord> pos,
		DArray<Coord> vel,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		pos[pId] += dt * vel[pId];
	}

	template<typename Real, typename Coord>
	__global__ void K_UpdatePosition(
		DArray<Coord> pos,
		DArray<Coord> vel,
		DArray<Attribute> att,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		Attribute att_i = att[pId];

		if (!att_i.isFixed())
		{
			pos[pId] += dt * vel[pId];
		}
	}

	template<typename TDataType>
	bool ParticleIntegrator<TDataType>::updatePosition()
	{
		Real dt = this->inTimeStep()->getData();

		int total_num = this->inPosition()->getDataPtr()->size();
		
		if(m_prePosition.size() != total_num)
			m_prePosition.resize(total_num);
		m_prePosition.assign(this->inPosition()->getData());

		if (this->inAttribute()->isEmpty())
		{
			cuExecute(total_num,
				K_UpdatePosition,
				this->inPosition()->getData(),
				this->inVelocity()->getData(),
				dt);
		}
		else
		{
			cuExecute(total_num,
				K_UpdatePosition,
				this->inPosition()->getData(),
				this->inVelocity()->getData(),
				this->inAttribute()->getData(),
				dt);
		}


		return true;
	}

	template<typename TDataType>
	bool ParticleIntegrator<TDataType>::integrate()
	{
		if (!this->inPosition()->isEmpty())
		{
			updatePosition();
			updateVelocity();
		}

		return true;
	}


	template<typename TDataType>
	void ParticleIntegrator<TDataType>::updateImpl()
	{
		this->begin();
		this->integrate();
		this->end();
	}



	DEFINE_CLASS(ParticleIntegrator);
}