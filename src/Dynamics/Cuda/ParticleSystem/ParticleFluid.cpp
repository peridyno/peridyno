#include "ParticleFluid.h"

//ParticleSystem
#include "Module/ParticleIntegrator.h"
#include "Module/ImplicitViscosity.h"
#include "Module/IterativeDensitySolver.h"

#include "ParticleSystemHelper.h"

//Framework
#include "Auxiliary/DataSource.h"

//Collision
#include "Collision/NeighborPointQuery.h"

//Topology
#include "Topology/PointSet.h"


namespace dyno
{
	IMPLEMENT_TCLASS(ParticleFluid, TDataType)

	template<typename TDataType>
	ParticleFluid<TDataType>::ParticleFluid()
		: ParticleSystem<TDataType>()
	{
		auto smoothingLength = this->animationPipeline()->createModule<FloatingNumber<TDataType>>();
		smoothingLength->varValue()->setValue(Real(0.006));

		auto samplingDistance = this->animationPipeline()->createModule<FloatingNumber<TDataType>>();
		samplingDistance->varValue()->setValue(Real(0.005));

		auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->stateTimeStep()->connect(integrator->inTimeStep());
		this->statePosition()->connect(integrator->inPosition());
		this->stateVelocity()->connect(integrator->inVelocity());
		this->stateForce()->connect(integrator->inForceDensity());
		this->animationPipeline()->pushModule(integrator);

		auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		smoothingLength->outFloating()->connect(nbrQuery->inRadius());
		this->statePosition()->connect(nbrQuery->inPosition());
		this->animationPipeline()->pushModule(nbrQuery);

		auto density = std::make_shared<IterativeDensitySolver<TDataType>>();
		smoothingLength->outFloating()->connect(density->inSmoothingLength());
		samplingDistance->outFloating()->connect(density->inSamplingDistance());
		this->stateTimeStep()->connect(density->inTimeStep());
		this->statePosition()->connect(density->inPosition());
		this->stateVelocity()->connect(density->inVelocity());
		nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
		this->animationPipeline()->pushModule(density);

		auto viscosity = std::make_shared<ImplicitViscosity<TDataType>>();
		viscosity->varViscosity()->setValue(Real(1.0));
		this->stateTimeStep()->connect(viscosity->inTimeStep());
		smoothingLength->outFloating()->connect(viscosity->inSmoothingLength());
		this->statePosition()->connect(viscosity->inPosition());
		this->stateVelocity()->connect(viscosity->inVelocity());
		nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
		this->animationPipeline()->pushModule(viscosity);
	}

	template<typename TDataType>
	ParticleFluid<TDataType>::~ParticleFluid()
	{
		Log::sendMessage(Log::Info, "ParticleFluid released \n");
	}

	template<typename TDataType>
	void ParticleFluid<TDataType>::preUpdateStates()
	{
		auto emitters = this->getParticleEmitters();

		int curNum = this->statePosition()->size();
		int totalNum = curNum;
		if (emitters.size() > 0)
		{
			for (int i = 0; i < emitters.size(); i++)
			{
				totalNum += emitters[i]->sizeOfParticles();
			}

			if (totalNum > curNum)
			{
				DArray<Coord> pBuf;
				DArray<Coord> vBuf;
				DArray<Coord> fBuf;

				if (curNum > 0)
				{
					pBuf.assign(this->statePosition()->getData());
					vBuf.assign(this->stateVelocity()->getData());
					fBuf.assign(this->stateForce()->getData());
				}

				this->statePosition()->resize(totalNum);
				this->stateVelocity()->resize(totalNum);

				//Currently, the force is simply set to zero
				this->stateForce()->resize(totalNum);
				this->stateForce()->reset();

				DArray<Coord>& new_pos = this->statePosition()->getData();
				DArray<Coord>& new_vel = this->stateVelocity()->getData();

				//Assign attributes from intial states
				if (curNum > 0)
				{
					new_pos.assign(pBuf, curNum, 0, 0);
					new_vel.assign(vBuf, curNum, 0, 0);

					pBuf.clear();
					vBuf.clear();
					fBuf.clear();
				}

				//Assign attributes from emitters
				int offset = curNum;
				for (int i = 0; i < emitters.size(); i++)
				{
					int num = emitters[i]->sizeOfParticles();
					if (num > 0)
					{
						DArray<Coord>& points = emitters[i]->getPositions();
						DArray<Coord>& vels = emitters[i]->getVelocities();

						new_pos.assign(points, num, offset, 0);
						new_vel.assign(vels, num, offset, 0);

						offset += num;
					}
				}
			}
		}

		if (this->varReshuffleParticles()->getValue())
		{
			auto& pos = this->statePosition()->getData();
			auto& vel = this->stateVelocity()->getData();
			auto& force = this->stateForce()->getData();

			DArray<OcKey> morton(pos.size());

			ParticleSystemHelper<TDataType>::calculateMortonCode(morton, pos, Real(0.005));
			ParticleSystemHelper<TDataType>::reorderParticles(pos, vel, force, morton);

			morton.clear();
		}
	}

	template<typename TDataType>
	void ParticleFluid<TDataType>::loadInitialStates()
	{
		auto initials = this->getInitialStates();

		if (initials.size() > 0) 
		{
			int totalNum = 0;

			for (int i = 0; i < initials.size(); i++)
			{
				totalNum += initials[i]->statePosition()->size();
			}

			this->statePosition()->resize(totalNum);
			this->stateVelocity()->resize(totalNum);
			this->stateForce()->resize(totalNum);
			this->stateForce()->reset();

			if (totalNum > 0)
			{
				DArray<Coord>& new_pos = this->statePosition()->getData();
				DArray<Coord>& new_vel = this->stateVelocity()->getData();
				DArray<Coord>& new_force = this->stateForce()->getData();

				int offset = 0;
				for (int i = 0; i < initials.size(); i++)
				{
					auto inPos = initials[i]->statePosition()->getDataPtr();
					auto inVel = initials[i]->stateVelocity()->getDataPtr();
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
			this->statePosition()->resize(0);
			this->stateVelocity()->resize(0);
			this->stateForce()->resize(0);
		}
	}

	template<typename TDataType>
	void ParticleFluid<TDataType>::resetStates()
	{
		loadInitialStates();

		if (!this->statePosition()->isEmpty())
		{
			auto points = this->statePointSet()->getDataPtr();
			points->setPoints(this->statePosition()->getData());
		}
		else
		{
			auto points = this->statePointSet()->getDataPtr();
			points->clear();
		}
	}

	template<typename TDataType>
	void ParticleFluid<TDataType>::reshuffleParticles()
	{

	}

	DEFINE_CLASS(ParticleFluid);
}