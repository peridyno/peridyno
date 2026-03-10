#include "ParticleFluid.h"

//ParticleSystem
#include "Module/CalculateNorm.h"
#include "Module/ParticleIntegrator.h"
#include "Module/ImplicitViscosity.h"

#include "Module/SemiImplicitDensitySolver.h"
#include "Module/IterativeDensitySolver.h"
#include "Module/DivergenceFreeSphSolver.h"
#include "Module/ImplicitISPH.h"

#include "ParticleSystemHelper.h"

//Framework
#include "Auxiliary/DataSource.h"

//Collision
#include "Collision/NeighborPointQuery.h"

//Topology
#include "Topology/PointSet.h"

//Rendering
#include "ColorMapping.h"
#include "GLPointVisualModule.h"

namespace dyno
{
	IMPLEMENT_TCLASS(ParticleFluid, TDataType)

	template<typename TDataType>
	ParticleFluid<TDataType>::ParticleFluid()
		: ParticleSystem<TDataType>()
	{
		this->varSamplingDistance()->setRange(0.001, 1);
		this->varSmoothingLength()->setRange(1, 3);

		auto callback = std::make_shared<FCallBackFunc>(
			[=]() {
				auto d = this->varSamplingDistance()->getValue();
				auto h = this->varSmoothingLength()->getValue();
				this->stateSamplingDistance()->setValue(d);
				this->stateSmoothingLength()->setValue(d * h);
			}
		);

		this->varSamplingDistance()->attach(callback);
		this->varSmoothingLength()->attach(callback);

		this->varSmoothingLength()->setValue(1.2);


		//declare a callback function for varSolverType();
		auto switchIncompressibilitySolver = std::make_shared <FCallBackFunc>(
			[&]() {
				auto setupSISPHSolver = [=] {
					this->animationPipeline()->clear();

					auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
					this->stateTimeStep()->connect(integrator->inTimeStep());
					this->statePosition()->connect(integrator->inPosition());
					this->stateVelocity()->connect(integrator->inVelocity());
					this->animationPipeline()->pushModule(integrator);

					auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
					this->stateSmoothingLength()->connect(nbrQuery->inRadius());
					this->statePosition()->connect(nbrQuery->inPosition());
					this->animationPipeline()->pushModule(nbrQuery);

					auto density = std::make_shared<SemiImplicitDensitySolver<TDataType>>();
					this->stateSamplingDistance()->connect(density->inSamplingDistance());
					this->stateSmoothingLength()->connect(density->inSmoothingLength());
					this->stateTimeStep()->connect(density->inTimeStep());
					this->statePosition()->connect(density->inPosition());
					this->stateVelocity()->connect(density->inVelocity());
					nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
					this->animationPipeline()->pushModule(density);

					auto viscosity = std::make_shared<ImplicitViscosity<TDataType>>();
					viscosity->varViscosity()->setValue(Real(1.0));
					this->stateTimeStep()->connect(viscosity->inTimeStep());
					this->stateSmoothingLength()->connect(viscosity->inSmoothingLength());
					this->stateSamplingDistance()->connect(viscosity->inSamplingDistance());
					this->statePosition()->connect(viscosity->inPosition());
					this->stateVelocity()->connect(viscosity->inVelocity());
					nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
					this->animationPipeline()->pushModule(viscosity);

					integrator->connect(density->importModules());
					density->connect(viscosity->importModules());
					};

				auto setupDFSPHSolver = [=] {
					this->animationPipeline()->clear();

					auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
					this->stateTimeStep()->connect(integrator->inTimeStep());
					this->statePosition()->connect(integrator->inPosition());
					this->stateVelocity()->connect(integrator->inVelocity());
					this->animationPipeline()->pushModule(integrator);

					auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
					this->stateSmoothingLength()->connect(nbrQuery->inRadius());
					this->statePosition()->connect(nbrQuery->inPosition());
					this->animationPipeline()->pushModule(nbrQuery);

					auto density = std::make_shared<DivergenceFreeSphSolver<DataType3f>>();
					density->varDivergenceSolverDisabled()->setValue(true);
					this->stateSmoothingLength()->connect(density->inSmoothingLength());
					this->stateSamplingDistance()->connect(density->inSamplingDistance());
					this->stateTimeStep()->connect(density->inTimeStep());
					this->statePosition()->connect(density->inPosition());
					this->stateVelocity()->connect(density->inVelocity());
					nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
					this->animationPipeline()->pushModule(density);

					auto viscosity = std::make_shared<ImplicitViscosity<TDataType>>();
					viscosity->varViscosity()->setValue(Real(1.0));
					this->stateTimeStep()->connect(viscosity->inTimeStep());
					this->stateSmoothingLength()->connect(viscosity->inSmoothingLength());
					this->stateSamplingDistance()->connect(viscosity->inSamplingDistance());
					this->statePosition()->connect(viscosity->inPosition());
					this->stateVelocity()->connect(viscosity->inVelocity());
					nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
					this->animationPipeline()->pushModule(viscosity);

					integrator->connect(density->importModules());
					density->connect(viscosity->importModules());
					};

				auto setupPBFSolver = [=] {
					this->animationPipeline()->clear();

					auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
					this->stateTimeStep()->connect(integrator->inTimeStep());
					this->statePosition()->connect(integrator->inPosition());
					this->stateVelocity()->connect(integrator->inVelocity());
					this->animationPipeline()->pushModule(integrator);

					auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
					this->stateSmoothingLength()->connect(nbrQuery->inRadius());
					this->statePosition()->connect(nbrQuery->inPosition());
					this->animationPipeline()->pushModule(nbrQuery);

					auto density = std::make_shared<IterativeDensitySolver<DataType3f>>();
					this->stateSmoothingLength()->connect(density->inSmoothingLength());
					this->stateSamplingDistance()->connect(density->inSamplingDistance());
					this->stateTimeStep()->connect(density->inTimeStep());
					this->statePosition()->connect(density->inPosition());
					this->stateVelocity()->connect(density->inVelocity());
					nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
					this->animationPipeline()->pushModule(density);

					auto viscosity = std::make_shared<ImplicitViscosity<TDataType>>();
					viscosity->varViscosity()->setValue(Real(1.0));
					this->stateTimeStep()->connect(viscosity->inTimeStep());
					this->stateSmoothingLength()->connect(viscosity->inSmoothingLength());
					this->stateSamplingDistance()->connect(viscosity->inSamplingDistance());
					this->statePosition()->connect(viscosity->inPosition());
					this->stateVelocity()->connect(viscosity->inVelocity());
					nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
					this->animationPipeline()->pushModule(viscosity);

					integrator->connect(density->importModules());
					density->connect(viscosity->importModules());
					};

				auto setupIISPHSolver = [=] {
					this->animationPipeline()->clear();

					auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
					this->stateTimeStep()->connect(integrator->inTimeStep());
					this->statePosition()->connect(integrator->inPosition());
					this->stateVelocity()->connect(integrator->inVelocity());
					this->animationPipeline()->pushModule(integrator);

					auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
					this->stateSmoothingLength()->connect(nbrQuery->inRadius());
					this->statePosition()->connect(nbrQuery->inPosition());
					this->animationPipeline()->pushModule(nbrQuery);

					auto density = std::make_shared<ImplicitISPH<DataType3f>>();
					this->stateSmoothingLength()->connect(density->inSmoothingLength());
					this->stateSamplingDistance()->connect(density->inSamplingDistance());
					this->stateTimeStep()->connect(density->inTimeStep());
					this->statePosition()->connect(density->inPosition());
					this->stateVelocity()->connect(density->inVelocity());
					nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
					this->animationPipeline()->pushModule(density);

					auto viscosity = std::make_shared<ImplicitViscosity<TDataType>>();
					viscosity->varViscosity()->setValue(Real(1.0));
					this->stateTimeStep()->connect(viscosity->inTimeStep());
					this->stateSmoothingLength()->connect(viscosity->inSmoothingLength());
					this->stateSamplingDistance()->connect(viscosity->inSamplingDistance());
					this->statePosition()->connect(viscosity->inPosition());
					this->stateVelocity()->connect(viscosity->inVelocity());
					nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
					this->animationPipeline()->pushModule(viscosity);

					integrator->connect(density->importModules());
					density->connect(viscosity->importModules());
					};

				auto k = this->varIncompressibilitySolver()->currentKey();
				switch (k)
				{
				case IncompressibilitySolver::SISPH:
					setupSISPHSolver();
					break;
				case IncompressibilitySolver::DFSPH:
					setupDFSPHSolver();
					break;
				case IncompressibilitySolver::PBF:
					setupPBFSolver();
					break;
				case IncompressibilitySolver::IISPH:
					setupIISPHSolver();
					break;
				default:
					break;
				}
			}
		);

		this->varIncompressibilitySolver()->attach(switchIncompressibilitySolver);
		this->varIncompressibilitySolver()->setCurrentKey(IncompressibilitySolver::SISPH);

		//Setup the default render modules
		auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
		this->stateVelocity()->connect(calculateNorm->inVec());
		this->graphicsPipeline()->pushModule(calculateNorm);

		auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
		colorMapper->varMax()->setValue(5.0f);
		calculateNorm->outNorm()->connect(colorMapper->inScalar());
		this->graphicsPipeline()->pushModule(colorMapper);

		auto ptRender = std::make_shared<GLPointVisualModule>();
		ptRender->varPointSize()->setValue(0.0035f);
		ptRender->setColor(Color(1, 0, 0));
		ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);

		this->statePointSet()->connect(ptRender->inPointSet());
		colorMapper->outColor()->connect(ptRender->inColor());

		this->graphicsPipeline()->pushModule(ptRender);

		this->setDt(Real(0.001));
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

				if (curNum > 0)
				{
					pBuf.assign(this->statePosition()->getData());
					vBuf.assign(this->stateVelocity()->getData());
				}

				this->statePosition()->resize(totalNum);
				this->stateVelocity()->resize(totalNum);

				DArray<Coord>& new_pos = this->statePosition()->getData();
				DArray<Coord>& new_vel = this->stateVelocity()->getData();

				//Assign attributes from initial states
				if (curNum > 0)
				{
					new_pos.assign(pBuf, curNum, 0, 0);
					new_vel.assign(vBuf, curNum, 0, 0);

					pBuf.clear();
					vBuf.clear();
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

			DArray<OcKey> morton(pos.size());

			ParticleSystemHelper<TDataType>::calculateMortonCode(morton, pos, Real(0.005));
			ParticleSystemHelper<TDataType>::reorderParticles(pos, vel, morton);

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

			if (totalNum > 0)
			{
				DArray<Coord>& new_pos = this->statePosition()->getData();
				DArray<Coord>& new_vel = this->stateVelocity()->getData();

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