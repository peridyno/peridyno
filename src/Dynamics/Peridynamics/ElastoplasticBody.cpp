#include "ElastoplasticBody.h"
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
#include "SharedFunc.h"


namespace dyno
{
	IMPLEMENT_CLASS_1(ElastoplasticBody, TDataType)

	template<typename TDataType>
	ElastoplasticBody<TDataType>::ElastoplasticBody(std::string name)
		: ParticleSystem<TDataType>(name)
	{
		m_horizon.setValue(0.0085);

		m_integrator = this->template setNumericalIntegrator<ParticleIntegrator<TDataType>>("integrator");
		this->currentPosition()->connect(m_integrator->inPosition());
		this->currentVelocity()->connect(m_integrator->inVelocity());
		this->currentForce()->connect(m_integrator->inForceDensity());
		this->animationPipeline()->pushModule(m_integrator);
		
		m_nbrQuery = this->template addComputeModule<NeighborPointQuery<TDataType>>("neighborhood");
		m_horizon.connect(m_nbrQuery->inRadius());
		this->currentPosition()->connect(m_nbrQuery->inPosition());
		this->animationPipeline()->pushModule(m_nbrQuery);

		m_plasticity = this->template addConstraintModule<ElastoplasticityModule<TDataType>>("elastoplasticity");
		m_horizon.connect(m_plasticity->inHorizon());
		this->varTimeStep()->connect(m_plasticity->inTimeStep());
		this->currentPosition()->connect(m_plasticity->inPosition());
		this->currentVelocity()->connect(m_plasticity->inVelocity());
		this->currentRestShape()->connect(m_plasticity->inRestShape());
		m_nbrQuery->outNeighborIds()->connect(m_plasticity->inNeighborIds());
		this->animationPipeline()->pushModule(m_plasticity);

		m_visModule = this->template addConstraintModule<ImplicitViscosity<TDataType>>("viscosity");
		m_visModule->varViscosity()->setValue(Real(1));
		m_horizon.connect(m_visModule->inSmoothingLength());
		this->currentPosition()->connect(m_visModule->inPosition());
		this->currentVelocity()->connect(m_visModule->inVelocity());
		m_nbrQuery->outNeighborIds()->connect(m_visModule->inNeighborIds());
		this->animationPipeline()->pushModule(m_visModule);

		m_surfaceNode = this->template createAncestor<Node>("Mesh");
		m_surfaceNode->setVisible(false);

		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		m_surfaceNode->setTopologyModule(triSet);

		std::shared_ptr<PointSetToPointSet<TDataType>> surfaceMapping = std::make_shared<PointSetToPointSet<TDataType>>(this->m_pSet, triSet);
		this->addTopologyMapping(surfaceMapping);
	}

	template<typename TDataType>
	ElastoplasticBody<TDataType>::~ElastoplasticBody()
	{
		
	}

	template<typename TDataType>
	void ElastoplasticBody<TDataType>::advance(Real dt)
	{
		this->animationPipeline()->update();
	}

	template<typename TDataType>
	bool ElastoplasticBody<TDataType>::resetStatus()
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
		}

		return true;
	}

	template<typename TDataType>
	void ElastoplasticBody<TDataType>::updateTopology()
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
	bool ElastoplasticBody<TDataType>::initialize()
	{
		m_nbrQuery->initialize();
		m_nbrQuery->compute();

		return ParticleSystem<TDataType>::initialize();
	}

	template<typename TDataType>
	void ElastoplasticBody<TDataType>::loadSurface(std::string filename)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->loadObjFile(filename);
	}

	template<typename TDataType>
	bool ElastoplasticBody<TDataType>::translate(Coord t)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->translate(t);

		return ParticleSystem<TDataType>::translate(t);
	}

	template<typename TDataType>
	bool ElastoplasticBody<TDataType>::scale(Real s)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->scale(s);

		return ParticleSystem<TDataType>::scale(s);
	}


	template<typename TDataType>
	void ElastoplasticBody<TDataType>::setElastoplasticitySolver(std::shared_ptr<ElastoplasticityModule<TDataType>> solver)
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

	DEFINE_CLASS(ElastoplasticBody);
}