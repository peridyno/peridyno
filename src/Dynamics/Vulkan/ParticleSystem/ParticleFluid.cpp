#include "ParticleFluid.h"

//ParticleSystem
#include "Module/ParticleIntegrator.h"
#include "Module/NeighborPointQuery.h"

//Framework
#include "Auxiliary/DataSource.h"

//Topology
#include "Topology/PointSet.h"

namespace dyno
{
	IMPLEMENT_CLASS(ParticleFluid)

	ParticleFluid::ParticleFluid()
		: ParticleSystem()
	{
		auto smoothingLength = std::make_shared<FloatingNumber<DataType3f>>();
		smoothingLength->varValue()->setValue(Real(0.006));
		this->animationPipeline()->pushModule(smoothingLength);

		auto integrator = std::make_shared<ParticleIntegrator>();
		this->stateTimeStep()->connect(integrator->inTimeStep());
		this->statePosition()->connect(integrator->inPosition());
		this->stateVelocity()->connect(integrator->inVelocity());
		this->animationPipeline()->pushModule(integrator);

		//TODO: replace other modules
// 		auto nbrQuery = std::make_shared<NeighborPointQuery>();
// 		smoothingLength->outFloating()->connect(nbrQuery->inRadius());
// 		this->statePosition()->connect(nbrQuery->inPosition());
// 		this->animationPipeline()->pushModule(nbrQuery);
// 
// 		auto density = std::make_shared<DensityPBD<TDataType>>();
// 		smoothingLength->outFloating()->connect(density->varSmoothingLength());
// 		this->stateTimeStep()->connect(density->inTimeStep());
// 		this->statePosition()->connect(density->inPosition());
// 		this->stateVelocity()->connect(density->inVelocity());
// 		nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
// 		this->animationPipeline()->pushModule(density);
// 
// 		auto viscosity = std::make_shared<ImplicitViscosity<TDataType>>();
// 		viscosity->varViscosity()->setValue(Real(1.0));
// 		this->stateTimeStep()->connect(viscosity->inTimeStep());
// 		smoothingLength->outFloating()->connect(viscosity->inSmoothingLength());
// 		this->statePosition()->connect(viscosity->inPosition());
// 		this->stateVelocity()->connect(viscosity->inVelocity());
// 		nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
// 		this->animationPipeline()->pushModule(viscosity);
	}

	ParticleFluid::~ParticleFluid()
	{
		Log::sendMessage(Log::Info, "ParticleFluid released \n");
	}

	void ParticleFluid::preUpdateStates()
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
				DArray<Vec3f> pBuf;
				DArray<Vec3f> vBuf;
				DArray<Vec3f> fBuf;

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

				DArray<Vec3f>& new_pos = this->statePosition()->getData();
				DArray<Vec3f>& new_vel = this->stateVelocity()->getData();

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
						DArray<Vec3f>& points = emitters[i]->getPositions();
						DArray<Vec3f>& vels = emitters[i]->getVelocities();

						new_pos.assign(points, num, offset, 0);
						new_vel.assign(vels, num, offset, 0);

						offset += num;
					}
				}
			}
 		}
	}

	void ParticleFluid::postUpdateStates()
	{
		if (!this->statePosition()->isEmpty())
		{
			auto ptSet = this->statePointSet()->getDataPtr();
			int num = this->statePosition()->size();
			auto& pts = ptSet->getPoints();
			if (num != pts.size())
			{
				pts.resize(num);
			}

			pts.assign(this->statePosition()->getData());
		}
	}

	void ParticleFluid::loadInitialStates()
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
				DArray<Vec3f>& new_pos = this->statePosition()->getData();
				DArray<Vec3f>& new_vel = this->stateVelocity()->getData();
				DArray<Vec3f>& new_force = this->stateForce()->getData();

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

	void ParticleFluid::resetStates()
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
}