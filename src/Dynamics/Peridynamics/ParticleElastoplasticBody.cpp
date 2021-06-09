#include "ParticleElastoplasticBody.h"
#include "ElastoplasticityModule.h"

#include "Topology/TriangleSet.h"
#include "Topology/PointSet.h"
#include "Peridynamics.h"
#include "Mapping/PointSetToPointSet.h"
#include "Topology/NeighborPointQuery.h"

#include "ParticleSystem/PositionBasedFluidModel.h"
#include "ParticleSystem/ParticleIntegrator.h"
#include "ParticleSystem/DensityPBD.h"
#include "ParticleSystem/ImplicitViscosity.h"


namespace dyno
{
	IMPLEMENT_CLASS_1(ParticleElastoplasticBody, TDataType)

	template<typename TDataType>
	ParticleElastoplasticBody<TDataType>::ParticleElastoplasticBody(std::string name)
		: ParticleSystem<TDataType>(name)
	{
		m_horizon.setValue(0.0085);

		m_integrator = this->template setNumericalIntegrator<ParticleIntegrator<TDataType>>("integrator");
		this->currentPosition()->connect(m_integrator->inPosition());
		this->currentVelocity()->connect(m_integrator->inVelocity());
		this->currentForce()->connect(m_integrator->inForceDensity());
		
		m_nbrQuery = this->template addComputeModule<NeighborPointQuery<TDataType>>("neighborhood");
		m_horizon.connect(m_nbrQuery->inRadius());
		this->currentPosition()->connect(m_nbrQuery->inPosition());

		m_plasticity = this->template addConstraintModule<ElastoplasticityModule<TDataType>>("elastoplasticity");
		m_horizon.connect(m_plasticity->inHorizon());
		this->currentPosition()->connect(m_plasticity->inPosition());
		this->currentVelocity()->connect(m_plasticity->inVelocity());
		m_nbrQuery->outNeighborIds()->connect(m_plasticity->inNeighborIds());

		m_pbdModule = this->template addConstraintModule<DensityPBD<TDataType>>("pbd");
		m_horizon.connect(m_pbdModule->varSmoothingLength());
		this->currentPosition()->connect(m_pbdModule->inPosition());
		this->currentVelocity()->connect(m_pbdModule->inVelocity());
		m_nbrQuery->outNeighborIds()->connect(m_pbdModule->inNeighborIds());

		m_visModule = this->template addConstraintModule<ImplicitViscosity<TDataType>>("viscosity");
		m_visModule->varViscosity()->setValue(Real(1));
		m_horizon.connect(m_visModule->inSmoothingLength());
		this->currentPosition()->connect(m_visModule->inPosition());
		this->currentVelocity()->connect(m_visModule->inVelocity());
		m_nbrQuery->outNeighborIds()->connect(m_visModule->inNeighborIds());


		m_surfaceNode = this->template createChild<Node>("Mesh");
		m_surfaceNode->setVisible(false);

		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		m_surfaceNode->setTopologyModule(triSet);

		std::shared_ptr<PointSetToPointSet<TDataType>> surfaceMapping = std::make_shared<PointSetToPointSet<TDataType>>(this->m_pSet, triSet);
		this->addTopologyMapping(surfaceMapping);
	}

	template<typename TDataType>
	ParticleElastoplasticBody<TDataType>::~ParticleElastoplasticBody()
	{
		
	}

	template<typename TDataType>
	void ParticleElastoplasticBody<TDataType>::advance(Real dt)
	{
		auto module = this->template getModule<ElastoplasticityModule<TDataType>>("elastoplasticity");

		m_integrator->begin();

		m_integrator->integrate();

		m_nbrQuery->compute();
		module->solveElasticity();
		m_nbrQuery->compute();

		module->applyPlasticity();

		m_visModule->constrain();

		m_integrator->end();
	}

	template<typename TDataType>
	void ParticleElastoplasticBody<TDataType>::updateTopology()
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
	bool ParticleElastoplasticBody<TDataType>::initialize()
	{
		m_nbrQuery->initialize();
		m_nbrQuery->compute();

		return ParticleSystem<TDataType>::initialize();
	}

	template<typename TDataType>
	void ParticleElastoplasticBody<TDataType>::loadSurface(std::string filename)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->loadObjFile(filename);
	}

	template<typename TDataType>
	bool ParticleElastoplasticBody<TDataType>::translate(Coord t)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->translate(t);

		return ParticleSystem<TDataType>::translate(t);
	}

	template<typename TDataType>
	bool ParticleElastoplasticBody<TDataType>::scale(Real s)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->scale(s);

		return ParticleSystem<TDataType>::scale(s);
	}


	template<typename TDataType>
	void ParticleElastoplasticBody<TDataType>::setElastoplasticitySolver(std::shared_ptr<ElastoplasticityModule<TDataType>> solver)
	{
		auto module = this->getModule("elastoplasticity");
		this->deleteModule(module);

		auto nbrQuery = this->template getModule<NeighborPointQuery<TDataType>>("neighborhood");

		this->currentPosition()->connect(solver->inPosition());
		this->currentVelocity()->connect(solver->inVelocity());
		nbrQuery->outNeighborIds()->connect(solver->inNeighborIds());
		m_horizon.connect(solver->inHorizon());

		solver->setName("elastoplasticity");
		this->addConstraintModule(solver);
	}

	DEFINE_CLASS(ParticleElastoplasticBody);
}