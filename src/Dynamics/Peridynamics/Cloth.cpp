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
	IMPLEMENT_CLASS_1(Cloth, TDataType)

	template<typename TDataType>
	Cloth<TDataType>::Cloth(std::string name)
		: ParticleSystem<TDataType>(name)
	{
		auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->currentPosition()->connect(integrator->inPosition());
		this->currentVelocity()->connect(integrator->inVelocity());
		this->currentForce()->connect(integrator->inForceDensity());

		this->animationPipeline()->pushModule(integrator);

		auto nbrQuery = this->template addComputeModule<NeighborPointQuery<TDataType>>("neighborhood");
		this->varHorizon()->connect(nbrQuery->inRadius());
		this->currentPosition()->connect(nbrQuery->inPosition());

		auto elasticity = std::make_shared<ElasticityModule<TDataType>>();
		this->varHorizon()->connect(elasticity->inHorizon());
		this->varTimeStep()->connect(elasticity->inTimeStep());
		this->currentPosition()->connect(elasticity->inPosition());
		this->currentVelocity()->connect(elasticity->inVelocity());
		this->currentRestShape()->connect(elasticity->inRestShape());
		nbrQuery->outNeighborIds()->connect(elasticity->inNeighborIds());
		this->animationPipeline()->pushModule(elasticity);


		auto fixed = std::make_shared<FixedPoints<TDataType>>();

		//Create a node for surface mesh rendering
		mSurfaceNode = this->template createAncestor<Node>("Mesh");

		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		mSurfaceNode->setTopologyModule(triSet);
	}

	template<typename TDataType>
	Cloth<TDataType>::~Cloth()
	{
		
	}

	template<typename TDataType>
	bool Cloth<TDataType>::translate(Coord t)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(mSurfaceNode->getTopologyModule())->translate(t);

		return ParticleSystem<TDataType>::translate(t);
	}


	template<typename TDataType>
	bool Cloth<TDataType>::scale(Real s)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(mSurfaceNode->getTopologyModule())->scale(s);

		return ParticleSystem<TDataType>::scale(s);
	}

	template<typename TDataType>
	void Cloth<TDataType>::updateTopology()
	{
		auto pts = this->m_pSet->getPoints();
		pts.assign(this->currentPosition()->getData());

		auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(mSurfaceNode->getTopologyModule());

		triSet->getPoints().assign(this->currentPosition()->getData());

		//TODO: topology mapping has bugs
// 		auto tMappings = this->getTopologyMappingList();
// 		for (auto iter = tMappings.begin(); iter != tMappings.end(); iter++)
// 		{
// 			(*iter)->apply();
// 		}
	}


	template<typename TDataType>
	void Cloth<TDataType>::resetStates()
	{
		ParticleSystem<TDataType>::resetStates();

		auto nbrQuery = this->template getModule<NeighborPointQuery<TDataType>>("neighborhood");
		nbrQuery->update();

		if (!this->currentPosition()->isEmpty())
		{
			this->currentRestShape()->allocate();
			auto nbrPtr = this->currentRestShape()->getDataPtr();
			nbrPtr->resize(nbrQuery->outNeighborIds()->getData());

			constructRestShape(*nbrPtr, nbrQuery->outNeighborIds()->getData(), this->currentPosition()->getData());
		}
	}

	template<typename TDataType>
	void Cloth<TDataType>::loadSurface(std::string filename)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(mSurfaceNode->getTopologyModule())->loadObjFile(filename);
	}

	template<typename TDataType>
	std::shared_ptr<Node> Cloth<TDataType>::getSurface()
	{
		return mSurfaceNode;
	}

	DEFINE_CLASS(Cloth);
}