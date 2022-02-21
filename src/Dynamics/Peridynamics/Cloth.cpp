#include "Cloth.h"
#include "Topology/TriangleSet.h"
#include "Topology/PointSet.h"
#include "Mapping/PointSetToPointSet.h"

#include "ParticleSystem/ParticleIntegrator.h"

#include "Topology/NeighborPointQuery.h"

#include "Peridynamics/ElasticityModule.h"
#include "Peridynamics/Peridynamics.h"
#include "Peridynamics/FixedPoints.h"

#include "SharedFunc.h"

namespace dyno
{
	IMPLEMENT_TCLASS(Cloth, TDataType)

	template<typename TDataType>
	Cloth<TDataType>::Cloth(std::string name)
		: ParticleSystem<TDataType>(name)
	{
		auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->varTimeStep()->connect(integrator->inTimeStep());
		this->statePosition()->connect(integrator->inPosition());
		this->stateVelocity()->connect(integrator->inVelocity());
		this->stateForce()->connect(integrator->inForceDensity());

		this->animationPipeline()->pushModule(integrator);

		auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		this->varHorizon()->connect(nbrQuery->inRadius());
		this->statePosition()->connect(nbrQuery->inPosition());
		this->animationPipeline()->pushModule(nbrQuery);

		auto elasticity = std::make_shared<ElasticityModule<TDataType>>();
		this->varHorizon()->connect(elasticity->inHorizon());
		this->varTimeStep()->connect(elasticity->inTimeStep());
		this->statePosition()->connect(elasticity->inPosition());
		this->stateVelocity()->connect(elasticity->inVelocity());
		this->currentRestShape()->connect(elasticity->inRestShape());
		nbrQuery->outNeighborIds()->connect(elasticity->inNeighborIds());
		this->animationPipeline()->pushModule(elasticity);


		auto fixed = std::make_shared<FixedPoints<TDataType>>();

		//Create a node for surface mesh rendering
		mSurfaceNode = std::make_shared<Node>("Mesh");
		mSurfaceNode->addAncestor(this);

		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		this->currentTopology()->setDataPtr(triSet);

		mSurfaceNode->currentTopology()->setDataPtr(triSet);
	}

	template<typename TDataType>
	Cloth<TDataType>::~Cloth()
	{
		
	}

	template<typename TDataType>
	bool Cloth<TDataType>::translate(Coord t)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(mSurfaceNode->currentTopology()->getDataPtr())->translate(t);

		return ParticleSystem<TDataType>::translate(t);
	}


	template<typename TDataType>
	bool Cloth<TDataType>::scale(Real s)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(mSurfaceNode->currentTopology()->getDataPtr())->scale(s);

		return ParticleSystem<TDataType>::scale(s);
	}

	template<typename TDataType>
	void Cloth<TDataType>::updateTopology()
	{
		auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->currentTopology()->getDataPtr());

		triSet->getPoints().assign(this->statePosition()->getData());
	}


	template<typename TDataType>
	void Cloth<TDataType>::resetStates()
	{
		ParticleSystem<TDataType>::resetStates();

		auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		this->varHorizon()->connect(nbrQuery->inRadius());
		this->statePosition()->connect(nbrQuery->inPosition());
		nbrQuery->update();

		if (!this->statePosition()->isEmpty())
		{
			this->currentRestShape()->allocate();
			auto nbrPtr = this->currentRestShape()->getDataPtr();
			nbrPtr->resize(nbrQuery->outNeighborIds()->getData());

			constructRestShape(*nbrPtr, nbrQuery->outNeighborIds()->getData(), this->statePosition()->getData());
		}
	}

	template<typename TDataType>
	void Cloth<TDataType>::loadSurface(std::string filename)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(mSurfaceNode->currentTopology()->getDataPtr())->loadObjFile(filename);
	}

	template<typename TDataType>
	std::shared_ptr<Node> Cloth<TDataType>::getSurface()
	{
		return mSurfaceNode;
	}

	DEFINE_CLASS(Cloth);
}