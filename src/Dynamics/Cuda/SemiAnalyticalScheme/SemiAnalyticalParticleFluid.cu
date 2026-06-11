#include "SemiAnalyticalParticleFluid.h"

//#include "PBFM.h"
#include "ParticleSystem/ParticleSystem.h"

#include "ParticleSystem/Module/ParticleIntegrator.h"
#include "ParticleSystem/Module/ImplicitViscosity.h"
#include "ParticleSystem/Module/SemiImplicitDensitySolver.h"

#include "Collision/NeighborPointQuery.h"
#include "Collision/DynamicNeighborTriangleQuery.h"
#include "Collision/NeighborTriangleQuery.h"

#include "ParticleSystem/Module//SurfaceEnergyForce.h"
#include "ParticleSystem/Module/ProjectionBasedFluidModel.h"

#include "Module/SemiAnalyticalDensitySolver.h"


namespace dyno
{
	//#define BROADSEARCH
//#define CFL_MODE
//#define DF_MODE
	IMPLEMENT_TCLASS(SemiAnalyticalParticleFluid, TDataType)
		template<typename TDataType>
	SemiAnalyticalParticleFluid<TDataType>::SemiAnalyticalParticleFluid()
		: ParticleFluid<TDataType>()
	{
		this->inTriangleSets()->tagOptional(true);

		//Clear the animation pipeline in ParticleFluid
		this->animationPipeline()->clear();

		//integrator
		auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
#ifdef CFL_MODE
		this->stateTimeStep_CFL()->connect(integrator->inTimeStep());
#else
		this->stateTimeStep()->connect(integrator->inTimeStep());
#endif // CFL_MODE
		this->statePosition()->connect(integrator->inPosition());
		this->stateVelocity()->connect(integrator->inVelocity());
		this->animationPipeline()->pushModule(integrator);

		//neighbor query
		auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		this->stateSmoothingLength()->connect(nbrQuery->inRadius());
		this->statePosition()->connect(nbrQuery->inPosition());
		this->animationPipeline()->pushModule(nbrQuery);

		//triangle neighbor
#ifdef BROADSEARCH
		auto nbrQueryTriMerge = std::make_shared<DynamicNeighborTriangleQuery<TDataType>>();
		this->stateSmoothingLength()->connect(nbrQueryTriMerge->inRadius());
#ifdef CFL_MODE
		this->stateTimeStep_CFL()->connect(nbrQueryTriMerge->inTimeStep());
#else
		this->stateTimeStep()->connect(nbrQueryTriMerge->inTimeStep());
#endif // CFL_MODE
		this->statePosition()->connect(nbrQueryTriMerge->inPosition());
		this->stateVelocity()->connect(nbrQueryTriMerge->inVelocity());
		this->inTriangleSets()->connect(nbrQueryTriMerge->inTriangleSet());
		this->statePreTriangleVertexMerge()->connect(nbrQueryTriMerge->inPreTriPosition());
		this->animationPipeline()->pushModule(nbrQueryTriMerge);
#else

		auto nbrQueryTriMerge = std::make_shared<NeighborTriangleQuery<TDataType>>();
		//this->varSearchRadius()->connect(nbrQueryTriMerge->inRadius());
		this->stateSmoothingLength()->connect(nbrQueryTriMerge->inRadius());
		this->statePosition()->connect(nbrQueryTriMerge->inPosition());
		this->inTriangleSets()->connect(nbrQueryTriMerge->inTriangleSet());
		this->animationPipeline()->pushModule(nbrQueryTriMerge);
#endif // BROADSEARCH
		//particle shifting
#ifdef DF_MODE
#else
		auto pshiftModule = std::make_shared<SemiAnalyticalDensitySolver<TDataType>>();
		this->stateSamplingDistance()->connect(pshiftModule->inSamplingDistance());
		this->stateSmoothingLength()->connect(pshiftModule->inSmoothingLength());
#ifdef CFL_MODE
		this->stateTimeStep_CFL()->connect(pshiftModule->inTimeStep());
#else
		this->stateTimeStep()->connect(pshiftModule->inTimeStep());
#endif // CFL_MODE
		this->statePosition()->connect(pshiftModule->inPosition());
		this->stateVelocity()->connect(pshiftModule->inVelocity());
		nbrQuery->outNeighborIds()->connect(pshiftModule->inNeighborIds());
		this->inTriangleSets()->connect(pshiftModule->inTriangleSetMerge());
		this->outDensity()->connect(pshiftModule->outDensity());
		this->outKappas()->connect(pshiftModule->outKappas());
		this->statePreTriangleVertexMerge()->connect(pshiftModule->inPreTriangleVerMerge());
		nbrQueryTriMerge->outNeighborIds()->connect(pshiftModule->inNeighborTriIdsMerge());
		this->animationPipeline()->pushModule(pshiftModule);
#endif // DF_MODE

		auto surfacetension = std::make_shared<SurfaceEnergyForce<DataType3f>>();
#ifdef CFL_MODE
		this->stateTimeStep_CFL()->connect(surfacetension->inTimeStep());
#else
		this->stateTimeStep()->connect(surfacetension->inTimeStep());
#endif // CFL_MODE
		this->stateSmoothingLength()->connect(surfacetension->inSmoothingLength());
		this->stateSamplingDistance()->connect(surfacetension->inSamplingDistance());
		this->statePosition()->connect(surfacetension->inPosition());
		this->stateVelocity()->connect(surfacetension->inVelocity());
		nbrQuery->outNeighborIds()->connect(surfacetension->inNeighborIds());
		this->animationPipeline()->pushModule(surfacetension);

		auto viscosity = std::make_shared<ImplicitViscosity<DataType3f>>();
#ifdef CFL_MODE
		this->stateTimeStep_CFL()->connect(viscosity->inTimeStep());
#else
		this->stateTimeStep()->connect(viscosity->inTimeStep());
#endif // CFL_MODE
		this->stateSmoothingLength()->connect(viscosity->inSmoothingLength());
		this->stateSamplingDistance()->connect(viscosity->inSamplingDistance());
		this->statePosition()->connect(viscosity->inPosition());
		this->stateVelocity()->connect(viscosity->inVelocity());
		nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
		this->animationPipeline()->pushModule(viscosity);

		integrator->connect(nbrQuery->importModules());
		integrator->connect(nbrQueryTriMerge->importModules());

		nbrQuery->connect(pshiftModule->importModules());
		nbrQueryTriMerge->connect(pshiftModule->importModules());

		pshiftModule->connect(surfacetension->importModules());
		surfacetension->connect(viscosity->importModules());
	}

	template<typename TDataType>
	SemiAnalyticalParticleFluid<TDataType>::~SemiAnalyticalParticleFluid()
	{

	}

	template<typename TDataType>
	bool SemiAnalyticalParticleFluid<TDataType>::validateInputs()
	{
		auto inBoundary = this->inTriangleSets()->getDataPtr();
		bool validateBoundary = inBoundary != nullptr && !inBoundary->isEmpty();

		bool ret = Node::validateInputs();

		return ret;
	}

	template<typename TDataType>
	void SemiAnalyticalParticleFluid<TDataType>::resetStates()
	{
		auto triSetMerge = this->inTriangleSets()->getDataPtr();
		if (triSetMerge != nullptr && !triSetMerge->isEmpty())
		{
			if (this->statePreTriangleVertexMerge()->getDataPtr() == nullptr)
			{
				this->statePreTriangleVertexMerge()->allocate();
			}
			this->statePreTriangleVertexMerge()->assign(triSetMerge->getPoints());
		}
		else
		{
			if (this->statePreTriangleVertexMerge()->getDataPtr() == nullptr)
			{
				this->statePreTriangleVertexMerge()->allocate();
			}
			this->statePreTriangleVertexMerge()->getData().clear();
		}
		this->stateTimeStep_CFL()->setValue(0.001f);
		this->stateSimulationTime()->setValue(0.0f);

		ParticleFluid<TDataType>::resetStates();
	}

	template<typename TDataType>
	void SemiAnalyticalParticleFluid<TDataType>::preUpdateStates()
	{
		ParticleFluid<TDataType>::preUpdateStates();

		if (this->statePosition()->getDataPtr() != nullptr)
		{
			auto& pos = this->statePosition()->getData();
			this->outDensity()->resize(pos.size());
			this->outKappas()->resize(pos.size());

		}

	}

	template<typename TDataType>
	void SemiAnalyticalParticleFluid<TDataType>::postUpdateStates()
	{

		ParticleFluid<TDataType>::postUpdateStates();
		auto triSetMerge = this->inTriangleSets()->getDataPtr();
		if (triSetMerge != nullptr && !triSetMerge->isEmpty())
		{
			if (this->statePreTriangleVertexMerge()->getDataPtr() == nullptr)
			{
				this->statePreTriangleVertexMerge()->allocate();
			}
			this->statePreTriangleVertexMerge()->assign(triSetMerge->getPoints());
		}
		else
		{
			// Ensure the state is allocated before accessing getData()
			// (getData() asserts when dataPtr is nullptr).
			if (this->statePreTriangleVertexMerge()->getDataPtr() == nullptr)
			{
				this->statePreTriangleVertexMerge()->allocate();
			}
			this->statePreTriangleVertexMerge()->getData().clear();
		}
#ifdef CFL_MODE
		this->stateSimulationTime()->setValue(this->stateSimulationTime()->getValue() + this->stateTimeStep_CFL()->getValue());
#else
		this->stateSimulationTime()->setValue(this->stateSimulationTime()->getValue() + this->stateTimeStep()->getValue());
#endif // CFL_MODE
		printf("Simulation Time: %f\n", this->stateSimulationTime()->getValue());
	}

	DEFINE_CLASS(SemiAnalyticalParticleFluid);
}
