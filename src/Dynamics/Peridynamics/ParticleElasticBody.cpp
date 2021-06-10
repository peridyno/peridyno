#include "ParticleElasticBody.h"
#include "Topology/TriangleSet.h"
#include "Topology/PointSet.h"
#include "Mapping/PointSetToPointSet.h"
#include "Topology/NeighborPointQuery.h"
#include "ParticleSystem/ParticleIntegrator.h"
#include "ElasticityModule.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(ParticleElasticBody, TDataType)

	template<typename TDataType>
	ParticleElasticBody<TDataType>::ParticleElasticBody(std::string name)
		: ParticleSystem<TDataType>(name)
	{
		this->varHorizon()->setValue(0.0085);
		//		this->attachField(&m_horizon, "horizon", "horizon");

		auto m_integrator = this->template setNumericalIntegrator<ParticleIntegrator<TDataType>>("integrator");
		this->currentPosition()->connect(m_integrator->inPosition());
		this->currentVelocity()->connect(m_integrator->inVelocity());
		this->currentForce()->connect(m_integrator->inForceDensity());

		this->getAnimationPipeline()->push_back(m_integrator);

		auto m_nbrQuery = this->template addComputeModule<NeighborPointQuery<TDataType>>("neighborhood");
		this->varHorizon()->connect(m_nbrQuery->inRadius());
		this->currentPosition()->connect(m_nbrQuery->inPosition());

		this->getAnimationPipeline()->push_back(m_nbrQuery);


		auto m_elasticity = this->template addConstraintModule<ElasticityModule<TDataType>>("elasticity");
		this->varHorizon()->connect(m_elasticity->inHorizon());
		this->currentPosition()->connect(m_elasticity->inPosition());
		this->currentVelocity()->connect(m_elasticity->inVelocity());
		m_nbrQuery->outNeighborIds()->connect(m_elasticity->inNeighborIds());

		this->getAnimationPipeline()->push_back(m_elasticity);

		//Create a node for surface mesh rendering
		m_surfaceNode = this->template createChild<Node>("Mesh");

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
		auto integrator = this->template getModule<ParticleIntegrator<TDataType>>("integrator");

		auto module = this->template getModule<ElasticityModule<TDataType>>("elasticity");

		integrator->begin();

		integrator->integrate();

		if (module != nullptr && self_update)
			module->constrain();

		integrator->end();
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
	std::shared_ptr<ElasticityModule<TDataType>> ParticleElasticBody<TDataType>::getElasticitySolver()
	{
		auto module = this->template getModule<ElasticityModule<TDataType>>("elasticity");
		return module;
	}


	template<typename TDataType>
	void ParticleElasticBody<TDataType>::setElasticitySolver(std::shared_ptr<ElasticityModule<TDataType>> solver)
	{
		auto nbrQuery = this->template getModule<NeighborPointQuery<TDataType>>("neighborhood");
		auto module = this->template getModule<ElasticityModule<TDataType>>("elasticity");

		this->currentPosition()->connect(solver->inPosition());
		this->currentVelocity()->connect(solver->inVelocity());
		nbrQuery->outNeighborIds()->connect(solver->inNeighborIds());
		this->varHorizon()->connect(solver->inHorizon());

		this->deleteModule(module);
		
		solver->setName("elasticity");
		this->addConstraintModule(solver);
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