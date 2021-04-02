#include "ParticleFluid.h"
#include "PositionBasedFluidModel.h"

#include "Topology/PointSet.h"
#include "SummationDensity.h"

#include <time.h>

namespace dyno
{
	IMPLEMENT_CLASS_1(ParticleFluid, TDataType)

	template<typename TDataType>
	ParticleFluid<TDataType>::ParticleFluid(std::string name)
		: ParticleSystem<TDataType>(name)
	{
		auto pbf = this->template setNumericalModel<PositionBasedFluidModel<TDataType>>("pbd");
		this->setNumericalModel(pbf);

		this->currentPosition()->connect(&pbf->m_position);
		this->currentVelocity()->connect(&pbf->m_velocity);
		this->currentForce()->connect(&pbf->m_forceDensity);
	}

	template<typename TDataType>
	ParticleFluid<TDataType>::~ParticleFluid()
	{
		
	}

	template<typename TDataType>
	void ParticleFluid<TDataType>::advance(Real dt)
	{		
		std::vector<std::shared_ptr<ParticleEmitter<TDataType>>> m_particleEmitters = this->getParticleEmitters();

		int total_num = 0;
		
		if (m_particleEmitters.size() > 0)
		{
			int total_num = this->currentPosition()->getElementCount();
			if (total_num > 0)
			{
				DArray<Coord>& position = this->currentPosition()->getValue();
				DArray<Coord>& velocity = this->currentVelocity()->getValue();
				DArray<Coord>& force = this->currentForce()->getValue();

				int start = 0;
				for (int i = 0; i < m_particleEmitters.size(); i++)
				{
					int num = m_particleEmitters[i]->currentPosition()->getElementCount();
					if (num > 0)
					{
						auto points = m_particleEmitters[i]->currentPosition()->getValue();
						auto vels = m_particleEmitters[i]->currentVelocity()->getValue();
						auto fors = m_particleEmitters[i]->currentForce()->getValue();

						cudaMemcpy(points.begin(), position.begin() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);
						cudaMemcpy(vels.begin(), velocity.begin() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);
						cudaMemcpy(fors.begin(), force.begin() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);
						start += num;
						// 						if (rand() % 1 == 0)
						// 							m_particleEmitters[i]->advance2(this->getDt());
					}
				}
			}
		}


		for (int i = 0; i < m_particleEmitters.size(); i++)
		{
			m_particleEmitters[i]->advance2(this->getDt());
		}

		total_num = 0;
		if (m_particleEmitters.size() > 0)
		{
			for (int i = 0; i < m_particleEmitters.size(); i++)
			{
				total_num += m_particleEmitters[i]->currentPosition()->getElementCount();
			}

			if (total_num > 0)
			{
				this->currentPosition()->setElementCount(total_num);
				this->currentVelocity()->setElementCount(total_num);
				this->currentForce()->setElementCount(total_num);

				//printf("###### %d\n", this->currentPosition()->getElementCount());

				DArray<Coord>& position = this->currentPosition()->getValue();
				DArray<Coord>& velocity = this->currentVelocity()->getValue();
				DArray<Coord>& force = this->currentForce()->getValue();

				int start = 0;
				for (int i = 0; i < m_particleEmitters.size(); i++)
				{
					int num = m_particleEmitters[i]->currentPosition()->getElementCount();
					if (num > 0)
					{
						DArray<Coord>& points = m_particleEmitters[i]->currentPosition()->getValue();
						DArray<Coord>& vels = m_particleEmitters[i]->currentVelocity()->getValue();
						DArray<Coord>& fors = m_particleEmitters[i]->currentForce()->getValue();

						cudaMemcpy(position.begin() + start, points.begin(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
						cudaMemcpy(velocity.begin() + start, vels.begin(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
						cudaMemcpy(force.begin() + start, fors.begin(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
						start += num;
					}
				}
			}
		}
		else
		{
			total_num = this->currentPosition()->getElementCount();
		}

		std::cout << "Total number: " << total_num << std::endl;

		if (total_num > 0 && this->self_update)
		{
			auto nModel = this->getNumericalModel();
			nModel->step(this->getDt());
		}

		//printf("%d\n", this->currentPosition()->getElementCount());

		
	}


	template<typename TDataType>
	bool ParticleFluid<TDataType>::resetStatus()
	{
		//printf("reset fluid\n");
		std::vector<std::shared_ptr<ParticleEmitter<TDataType>>> m_particleEmitters = this->getParticleEmitters();
		if(m_particleEmitters.size() > 0)
		{ 
			this->currentPosition()->setElementCount(0);
			this->currentVelocity()->setElementCount(0);
			this->currentForce()->setElementCount(0);
		}
		else 
			return ParticleSystem<TDataType>::resetStatus();
		return true;
	}

	DEFINE_CLASS(ParticleFluid);
}