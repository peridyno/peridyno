#include <cuda_runtime.h>
#include "ParticleIntegrator.h"
#include "Framework/FieldArray.h"
#include "Framework/FieldVar.h"
#include "Framework/Node.h"
#include "Utility.h"
#include "Framework/SceneGraph.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(ParticleIntegrator, TDataType)

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
			int num = this->inPosition()->getElementCount();
			
			m_prePosition.resize(num);
			m_preVelocity.resize(num);

			Function1Pt::copy(m_prePosition, this->inPosition()->getValue());
			Function1Pt::copy(m_preVelocity, this->inVelocity()->getValue());

			this->inForceDensity()->getReference()->reset();
		}
	}

	template<typename TDataType>
	void ParticleIntegrator<TDataType>::end()
	{

	}

	template<typename TDataType>
	bool ParticleIntegrator<TDataType>::initializeImpl()
	{
// 		if (!isAllFieldsReady())
// 		{
// 			std::cout << "Exception: " << std::string("DensitySummation's fields are not fully initialized!") << "\n";
// 			return false;
// 		}
// 
// 		int num = this->inPosition()->getElementCount();
// 
// 		m_prePosition.resize(num);
// 		m_preVelocity.resize(num);

		return true;
	}

	template<typename Real, typename Coord>
	__global__ void K_UpdateVelocity(
		DeviceArray<Coord> vel,
		DeviceArray<Coord> forceDensity,
		Coord gravity,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= forceDensity.size()) return;

		vel[pId] += dt * (forceDensity[pId] + gravity);
	}


	template<typename Real, typename Coord>
	__global__ void K_UpdateVelocity(
		DeviceArray<Coord> vel,
		DeviceArray<Coord> forceDensity,
		DeviceArray<Attribute> atts,
		Coord gravity,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= forceDensity.size()) return;

		Attribute att = atts[pId];

		if (att.IsDynamic())
		{
			vel[pId] += dt * (forceDensity[pId] + gravity);
		}
	}

	template<typename TDataType>
	bool ParticleIntegrator<TDataType>::updateVelocity()
	{
		Real dt = 0.001;
		if(this->getParent() != NULL)
			dt = getParent()->getDt();
		Coord gravity = SceneGraph::getInstance().getGravity();

		int total_num = this->inPosition()->getElementCount();

		if (this->inAttribute()->isEmpty())
		{
			cuExecute(total_num,
				K_UpdateVelocity,
				this->inVelocity()->getValue(),
				this->inForceDensity()->getValue(),
				gravity,
				dt);
		}
		else
		{
			cuExecute(total_num,
				K_UpdateVelocity,
				this->inVelocity()->getValue(),
				this->inForceDensity()->getValue(),
				this->inAttribute()->getValue(),
				gravity,
				dt);
		}
		

		return true;
	}

	template<typename Real, typename Coord>
	__global__ void K_UpdatePosition(
		DeviceArray<Coord> pos,
		DeviceArray<Coord> vel,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		pos[pId] += dt * vel[pId];
	}

	template<typename Real, typename Coord>
	__global__ void K_UpdatePosition(
		DeviceArray<Coord> pos,
		DeviceArray<Coord> vel,
		DeviceArray<Attribute> att,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		Attribute att_i = att[pId];

		if (!att_i.IsFixed())
		{
			pos[pId] += dt * vel[pId];
		}
	}

	template<typename TDataType>
	bool ParticleIntegrator<TDataType>::updatePosition()
	{
		Real dt = 0.001;
		if (this->getParent() != NULL)
			dt = getParent()->getDt();

		int total_num = this->inPosition()->getReference()->size();
		if (this->inAttribute()->isEmpty())
		{
			cuExecute(total_num,
				K_UpdatePosition,
				this->inPosition()->getValue(),
				this->inVelocity()->getValue(),
				dt);
		}
		else
		{
			cuExecute(total_num,
				K_UpdatePosition,
				this->inPosition()->getValue(),
				this->inVelocity()->getValue(),
				this->inAttribute()->getValue(),
				dt);
		}


		return true;
	}

	template<typename TDataType>
	bool ParticleIntegrator<TDataType>::integrate()
	{
		if (!this->inPosition()->isEmpty())
		{
			updateVelocity();
			updatePosition();
		}

		return true;
	}
}