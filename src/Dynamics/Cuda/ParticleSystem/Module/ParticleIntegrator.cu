#include <cuda_runtime.h>
#include "ParticleIntegrator.h"
#include "Node.h"
#include "SceneGraphFactory.h"

namespace dyno
{
	IMPLEMENT_TCLASS(ParticleIntegrator, TDataType)

	template<typename TDataType>
	ParticleIntegrator<TDataType>::ParticleIntegrator()
		: ComputeModule()
	{
		this->inAttribute()->tagOptional(true);
	}

	template<typename Real, typename Coord>
	__global__ void K_UpdateVelocity(
		DArray<Coord> vel,
		Coord gravity,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vel.size()) return;

		vel[pId] += dt * (gravity);
	}


	

	template<typename Real, typename Coord>
	__global__ void K_UpdateVelocity(
		DArray<Coord> vel,
		DArray<Attribute> atts,
		Coord gravity,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vel.size()) return;

		Attribute att = atts[pId];

		if (att.isDynamic())
		{
			vel[pId] += dt * (gravity);
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
				gravity,
				dt);
		}
		else
		{
			cuExecute(total_num,
				K_UpdateVelocity,
				this->inVelocity()->getData(),
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
	void ParticleIntegrator<TDataType>::compute()
	{
		updatePosition();
		updateVelocity();
	}

	DEFINE_CLASS(ParticleIntegrator);
}