#include "Peridynamics.h"

#include "Collision/NeighborPointQuery.h"

#include "SharedFunc.h"

namespace dyno
{
	template<typename TDataType>
	Peridynamics<TDataType>::Peridynamics()
		: ParticleSystem<TDataType>()
	{
		this->varHorizon()->attach(
			std::make_shared<FCallBackFunc>(
				[=]() {
					this->stateHorizon()->setValue(this->varHorizon()->getValue());
				})
		);

		this->varHorizon()->setValue(0.0085);

		this->setDt(0.001f);
	}

	template<typename TDataType>
	Peridynamics<TDataType>::~Peridynamics()
	{
		
	}

	template<typename TDataType>
	std::string Peridynamics<TDataType>::getNodeType()
	{
		return "Peridynamics";
	}

	template<typename TDataType>
	void Peridynamics<TDataType>::resetStates()
	{
		loadSolidParticles();

		auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
 		this->stateHorizon()->connect(nbrQuery->inRadius());
 		this->statePosition()->connect(nbrQuery->inPosition());
		nbrQuery->update();

		if (!this->statePosition()->isEmpty())
		{
			this->stateBonds()->allocate();
			auto nbrPtr = this->stateBonds()->getDataPtr();
			nbrPtr->resize(nbrQuery->outNeighborIds()->getData());

			constructRestShape(*nbrPtr, nbrQuery->outNeighborIds()->getData(), this->statePosition()->getData());

			this->stateReferencePosition()->assign(this->statePosition()->getData());
		}

		ParticleSystem<TDataType>::resetStates();
	}

	template<typename TDataType>
	void Peridynamics<TDataType>::loadSolidParticles()
	{
		auto particles = this->getSolidParticles();

		//Merge solid particles
		if (particles.size() > 0)
		{
			int totalNum = 0;

			for (int i = 0; i < particles.size(); i++)
			{
				totalNum += particles[i]->statePosition()->size();
			}

			this->statePosition()->resize(totalNum);
			this->stateVelocity()->resize(totalNum);

			if (totalNum > 0)
			{
				DArray<Coord>& new_pos = this->statePosition()->getData();
				DArray<Coord>& new_vel = this->stateVelocity()->getData();

				int offset = 0;
				for (int i = 0; i < particles.size(); i++)
				{
					auto inPos = particles[i]->statePosition()->getDataPtr();
					auto inVel = particles[i]->stateVelocity()->getDataPtr();
					if (!inPos->isEmpty())
					{
						uint num = inPos->size();

						new_pos.assign(*inPos, num, offset, 0);
						new_vel.assign(*inVel, num, offset, 0);

						offset += num;
					}
				}
			}
		}
		else {
			this->statePosition()->clear();
			this->stateVelocity()->clear();
			this->stateReferencePosition()->clear();
			this->stateBonds()->clear();
		}
	}

	DEFINE_CLASS(Peridynamics);
}