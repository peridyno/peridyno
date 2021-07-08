#include "ParticleElasticBody.h"
#include "Topology/TriangleSet.h"
#include "Topology/PointSet.h"
#include "Mapping/PointSetToPointSet.h"
#include "Topology/NeighborPointQuery.h"
#include "ParticleSystem/ParticleIntegrator.h"
#include "ElasticityModule.h"
#include "SharedFunc.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(ParticleElasticBody, TDataType)

	template<typename TDataType>
	ParticleElasticBody<TDataType>::ParticleElasticBody(std::string name)
		: ParticleSystem<TDataType>(name)
	{
		this->varHorizon()->setValue(0.0085);
//		this->attachField(&m_horizon, "horizon", "horizon");

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
		this->currentPosition()->connect(elasticity->inPosition());
		this->currentVelocity()->connect(elasticity->inVelocity());
		this->currentRestShape()->connect(elasticity->inRestShape());
		nbrQuery->outNeighborIds()->connect(elasticity->inNeighborIds());

		this->animationPipeline()->pushModule(elasticity);

		//Create a node for surface mesh rendering
		m_surfaceNode = this->template createAncestor<Node>("Mesh");

		auto triSet = m_surfaceNode->template setTopologyModule<TriangleSet<TDataType>>("surface_mesh");

		//Set the topology mapping from PointSet to TriangleSet
		auto surfaceMapping = this->template addTopologyMapping<PointSetToPointSet<TDataType>>("surface_mapping");
		surfaceMapping->setFrom(this->m_pSet);
		surfaceMapping->setTo(triSet);
	}

	template<typename TDataType>
	ParticleElasticBody<TDataType>::~ParticleElasticBody()
	{
		
	}

	template<typename TDataType>
	bool ParticleElasticBody<TDataType>::translate(Coord t)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->translate(t);

		return ParticleSystem<TDataType>::translate(t);
	}

	template<typename TDataType>
	bool ParticleElasticBody<TDataType>::scale(Real s)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->scale(s);

		return ParticleSystem<TDataType>::scale(s);
	}


	template<typename TDataType>
	bool ParticleElasticBody<TDataType>::initialize()
	{
		return ParticleSystem<TDataType>::initialize();
	}

	template<typename TDataType>
	void ParticleElasticBody<TDataType>::advance(Real dt)
	{
		this->animationPipeline()->update();
	}

	template<typename TDataType>
	void ParticleElasticBody<TDataType>::updateTopology()
	{
		auto pts = this->m_pSet->getPoints();
		pts.assign(this->currentPosition()->getData());

		auto tMappings = this->getTopologyMappingList();
		for (auto iter = tMappings.begin(); iter != tMappings.end(); iter++)
		{
			(*iter)->apply();
		}
	}

	template<typename TDataType>
	bool ParticleElasticBody<TDataType>::resetStatus()
	{
		ParticleSystem<TDataType>::resetStatus();

		auto nbrQuery = this->template getModule<NeighborPointQuery<TDataType>>("neighborhood");
		nbrQuery->update();

		if (!this->currentPosition()->isEmpty())
		{
			this->currentRestShape()->allocate();
			auto nbrPtr = this->currentRestShape()->getDataPtr();
			nbrPtr->resize(nbrQuery->outNeighborIds()->getData());

			constructRestShape(*nbrPtr, nbrQuery->outNeighborIds()->getData(), this->currentPosition()->getData());

			this->currentReferencePosition()->allocate();
			this->currentReferencePosition()->getDataPtr()->assign(this->currentPosition()->getData());

			this->currentNeighborIds()->allocate();
			this->currentNeighborIds()->getDataPtr()->assign(nbrQuery->outNeighborIds()->getData());
		}

		return true;
	}

	template<typename TDataType>
	void ParticleElasticBody<TDataType>::loadSurface(std::string filename)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->loadObjFile(filename);
	}


	template<typename TDataType>
	std::shared_ptr<PointSetToPointSet<TDataType>> ParticleElasticBody<TDataType>::getTopologyMapping()
	{
		auto mapping = this->template getModule<PointSetToPointSet<TDataType>>("surface_mapping");

		return mapping;
	}

	DEFINE_CLASS(ParticleElasticBody);
}