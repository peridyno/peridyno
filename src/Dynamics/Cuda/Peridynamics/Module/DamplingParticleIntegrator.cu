#include <cuda_runtime.h>
#include "DamplingParticleIntegrator.h"
#include "Node.h"
#include "SceneGraphFactory.h"

namespace dyno
{
	//IMPLEMENT_TCLASS(DamplingParticleIntegrator, TDataType)

	template<typename TDataType>
	DamplingParticleIntegrator<TDataType>::DamplingParticleIntegrator()
		:NumericalIntegrator()
	{
		this->inAttribute()->tagOptional(true);
	}

	template<typename TDataType>
	void DamplingParticleIntegrator<TDataType>::begin()
	{
		if (!this->inPosition()->isEmpty())
			this->inForceDensity()->getDataPtr()->reset();
	}


	template<typename Real, typename Coord>
	__global__ void K_Disspation_Velocity(
		DArray<Coord> vel,
		Real disspation)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vel.size()) return;

		vel[pId] *= disspation;
	}

	template<typename TDataType>
	void DamplingParticleIntegrator<TDataType>::end()
	{
		int vNum = this->inPosition()->getData().size();
		
		cuExecute(vNum,
			K_Disspation_Velocity,
			this->inVelocity()->getData(),
			this->inAirDisspation()->getData());
	}

	template<typename Real, typename Coord>
	__global__ void K_UpdateVelocity(
		DArray<Coord> vel,
		DArray<Coord> forceDensity,
		DArray<Coord> contactForce,
		DArray<Coord> Norm,
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
		DArray<Coord> pos,
		DArray<Coord> forceDensity,
		DArray<Coord> contactForce,
		DArray<Coord> Norm,
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
	__global__ void K_Friction(
		DArray<Coord> Norm,
		DArray<Coord> Force,
		DArray<Coord> Velocity,
		DArray<Coord> ForceDensity,
		Real mu,
		Coord g,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Norm.size()) return;
		
		Coord f = Force[pId] + g + ForceDensity[pId];
		if (f.dot(Norm[pId]) <= EPSILON)//enforced
		{
			Coord f_n = f.dot(Norm[pId]) * Norm[pId];
			Coord f_t = f - f_n;
			Coord v_n = Velocity[pId].dot(Norm[pId]) * Norm[pId];
			Coord v_t = Velocity[pId] - v_n;
			if (v_t.norm() <= EPSILON)
				return;
			Real damp;
			if (f_t.norm() >= mu * f_n.norm()) //kinetic friction
			{
				damp = max(0.0, 1.0 - f_n.norm() * mu * dt / v_t.norm());
			}
			else { //static friction
				damp = 0.0;
			}
			Velocity[pId] = v_n + v_t * damp;
		}

	}

	template<typename Real, typename Coord>
	__global__ void K_Friction(
		DArray<Coord> Norm,
		DArray<Coord> Force,
		DArray<Coord> Velocity,
		DArray<Coord> ForceDensity,
		DArray<Attribute> atts,
		Real mu,
		Coord g,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Norm.size()) return;

		Attribute att = atts[pId];

		if (att.isDynamic())
		{

			Coord f = Force[pId] + g + ForceDensity[pId];
			if (f.dot(Norm[pId]) <= EPSILON)//enforced
			{
				Coord f_n = f.dot(Norm[pId]) * Norm[pId];
				Coord f_t = f - f_n;
				Coord v_n = Velocity[pId].dot(Norm[pId]) * Norm[pId];
				Coord v_t = Velocity[pId] - v_n;
				if (v_t.norm() <= EPSILON)
					return;
				Real damp;
				if (f_t.norm() >= mu * f_n.norm()) //kinetic friction
				{
					damp = max(0.0, 1.0 - f_n.norm() * mu * dt / v_t.norm());
				}
				else { //static friction
					damp = 0.0;
				}
				Velocity[pId] = v_n + v_t * damp;
			}
		}
	}


	template<typename TDataType>
	bool DamplingParticleIntegrator<TDataType>::updateVelocity()
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
				this->inContactForce()->getData(),
				this->inNorm()->getData(),
				gravity,
				dt);

			if (this->inMu()->getData() >= EPSILON) //Columbian friction model
			{
				cuExecute(total_num,
					K_Friction,
					this->inNorm()->getData(),
					this->inContactForce()->getData(),
					this->inVelocity()->getData(),
					this->inForceDensity()->getData(),
					this->inMu()->getData(),
					gravity,
					dt);
				

				cuSynchronize();
			}
		}
		else//with att
		{
			
			
			
			cuExecute(total_num,
				K_UpdateVelocity,
				this->inVelocity()->getData(),
				this->inPosition()->getData(),
				this->inForceDensity()->getData(),
				this->inContactForce()->getData(),
				this->inNorm()->getData(),
				this->inAttribute()->getData(),
				gravity,
				dt);

			if (this->inMu()->getData() >= EPSILON) //Columbian friction model
			{
				cuExecute(total_num,
					K_Friction,
					this->inNorm()->getData(),
					this->inContactForce()->getData(),
					this->inVelocity()->getData(),
					this->inForceDensity()->getData(),
					this->inAttribute()->getData(),
					this->inMu()->getData(),
					gravity,
					dt);

				cuSynchronize();

			}
			
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
		Coord po = pos[pId];
		if (!att_i.isFixed())
		{
			pos[pId] += dt * vel[pId];
		}
	}

	template<typename TDataType>
	bool DamplingParticleIntegrator<TDataType>::updatePosition()
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
	bool DamplingParticleIntegrator<TDataType>::integrate()
	{
		if (!this->inPosition()->isEmpty())
		{
			updateVelocity();
			updatePosition();
			
		}

		return true;
	}


	template<typename TDataType>
	void DamplingParticleIntegrator<TDataType>::updateImpl()
	{
		this->begin();
		this->integrate();
		this->end();
	}



	DEFINE_CLASS(DamplingParticleIntegrator);
}