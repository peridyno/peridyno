#include "SemiAnalyticalSFINode.h"

#include "Module/SemiAnalyticalSharedFunc.h"

//#include "PBFM.h"
#include "Module/SemiAnalyticalPositionBasedFluidModel.h"
#include "Module/SemiAnalyticalParticleShifting.h"
#include "Module/TriangularMeshConstraint.h"

#include "ParticleSystem/ParticleSystem.h"

#include "ParticleSystem/Module/ParticleIntegrator.h"
#include "ParticleSystem/Module/ImplicitViscosity.h"

#include "Collision/NeighborPointQuery.h"
#include "Collision/NeighborTriangleQuery.h"

namespace dyno
{
	IMPLEMENT_TCLASS(SemiAnalyticalSFINode, TDataType)

	template<typename TDataType>
	SemiAnalyticalSFINode<TDataType>::SemiAnalyticalSFINode()
		: ParticleFluid<TDataType>()
	{
		//Clear the animation pipeline in ParticleFluid
		this->animationPipeline()->clear();

		this->varSamplingDistance()->setValue(0.005f);
		this->varSmoothingLength()->setValue(2.4f);

		//integrator
		auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->stateTimeStep()->connect(integrator->inTimeStep());
		this->statePosition()->connect(integrator->inPosition());
		this->stateVelocity()->connect(integrator->inVelocity());
		this->animationPipeline()->pushModule(integrator);

		//neighbor query
		auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		this->stateSmoothingLength()->connect(nbrQuery->inRadius());
		this->statePosition()->connect(nbrQuery->inPosition());
		this->animationPipeline()->pushModule(nbrQuery);

		//triangle neighbor
		auto nbrQueryTri = std::make_shared<NeighborTriangleQuery<TDataType>>();
		this->stateSmoothingLength()->connect(nbrQueryTri->inRadius());
		this->statePosition()->connect(nbrQueryTri->inPosition());
		this->inTriangleSet()->connect(nbrQueryTri->inTriangleSet());
		this->animationPipeline()->pushModule(nbrQueryTri);

		//mesh collision
		auto meshCollision = std::make_shared<TriangularMeshConstraint<TDataType>>();
		this->stateTimeStep()->connect(meshCollision->inTimeStep());
		this->statePosition()->connect(meshCollision->inPosition());
		this->stateVelocity()->connect(meshCollision->inVelocity());
// 		this->stateTriangleVertex()->connect(meshCollision->inTriangleVertex());
// 		this->stateTriangleIndex()->connect(meshCollision->inTriangleIndex());
		this->inTriangleSet()->connect(meshCollision->inTriangleSet());
		nbrQueryTri->outNeighborIds()->connect(meshCollision->inTriangleNeighborIds());
		this->animationPipeline()->pushModule(meshCollision);

		//viscosity
		auto viscosity = std::make_shared<ImplicitViscosity<TDataType>>();
		viscosity->varViscosity()->setValue(Real(0.5));//0.5
		this->stateTimeStep()->connect(viscosity->inTimeStep());
		this->stateSmoothingLength()->connect(viscosity->inSmoothingLength());
		this->stateSamplingDistance()->connect(viscosity->inSamplingDistance());
		this->statePosition()->connect(viscosity->inPosition());
		this->stateVelocity()->connect(viscosity->inVelocity());
		nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
		this->animationPipeline()->pushModule(viscosity);

		//particle shifting
		auto pshiftModule = std::make_shared<SemiAnalyticalParticleShifting<TDataType>>();
		this->stateSamplingDistance()->connect(pshiftModule->inSamplingDistance());
		this->stateSmoothingLength()->connect(pshiftModule->inSmoothingLength());
		this->stateTimeStep()->connect(pshiftModule->inTimeStep());
		this->statePosition()->connect(pshiftModule->inPosition());
		this->stateVelocity()->connect(pshiftModule->inVelocity());
		this->inTriangleSet()->connect(pshiftModule->inTriangleSet());
		nbrQuery->outNeighborIds()->connect(pshiftModule->inNeighborIds());
		nbrQueryTri->outNeighborIds()->connect(pshiftModule->inNeighborTriIds());
		this->animationPipeline()->pushModule(pshiftModule);

		this->setDt(0.001f);
	}

	template<typename TDataType>
	SemiAnalyticalSFINode<TDataType>::~SemiAnalyticalSFINode()
	{
	
	}

	template<typename TDataType>
	bool SemiAnalyticalSFINode<TDataType>::validateInputs()
	{
		auto inBoundary = this->inTriangleSet()->getDataPtr();
		bool validateBoundary = inBoundary != nullptr && !inBoundary->isEmpty();

		bool ret = Node::validateInputs();

		return ret && validateBoundary;
	}

	template<typename TDataType>
	void SemiAnalyticalSFINode<TDataType>::resetStates()
	{
		if (this->varFast()->getData() == true)
		{
			this->animationPipeline()->clear();
			auto pbd = std::make_shared<SemiAnalyticalPositionBasedFluidModel<DataType3f>>();
			pbd->varSmoothingLength()->setValue(0.0085);

			this->animationPipeline()->clear();
			this->stateTimeStep()->connect(pbd->inTimeStep());
			this->statePosition()->connect(pbd->inPosition());
			this->stateVelocity()->connect(pbd->inVelocity());
			this->inTriangleSet()->connect(pbd->inTriangleSet());
			this->animationPipeline()->pushModule(pbd);
		}

		ParticleFluid<TDataType>::resetStates();
	}

	template<typename TDataType>
	void SemiAnalyticalSFINode<TDataType>::preUpdateStates()
	{
		ParticleFluid<TDataType>::preUpdateStates();
	}

	template<typename TDataType>
	void SemiAnalyticalSFINode<TDataType>::postUpdateStates()
	{
		ParticleFluid<TDataType>::postUpdateStates();
	}

	DEFINE_CLASS(SemiAnalyticalSFINode);
}