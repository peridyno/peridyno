#include "ElastoplasticBody.h"

#include "Topology/TriangleSet.h"
#include "Topology/PointSet.h"
#include "Topology/NeighborPointQuery.h"

#include "Mapping/PointSetToPointSet.h"

#include "Module/Peridynamics.h"
#include "Module/ElastoplasticityModule.h"

#include "ParticleSystem/PositionBasedFluidModel.h"
#include "ParticleSystem/ParticleIntegrator.h"
#include "ParticleSystem/DensityPBD.h"
#include "ParticleSystem/ImplicitViscosity.h"
#include "SharedFunc.h"


namespace dyno
{
	IMPLEMENT_TCLASS(ElastoplasticBody, TDataType)

	template<typename TDataType>
	ElastoplasticBody<TDataType>::ElastoplasticBody(std::string name)
		: ParticleSystem<TDataType>(name)
	{
		m_horizon.setValue(0.0085);

		m_integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->varTimeStep()->connect(m_integrator->inTimeStep());
		this->statePosition()->connect(m_integrator->inPosition());
		this->stateVelocity()->connect(m_integrator->inVelocity());
		this->stateForce()->connect(m_integrator->inForceDensity());
		this->animationPipeline()->pushModule(m_integrator);
		
		m_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		m_horizon.connect(m_nbrQuery->inRadius());
		this->statePosition()->connect(m_nbrQuery->inPosition());
		this->animationPipeline()->pushModule(m_nbrQuery);

		m_plasticity = std::make_shared<ElastoplasticityModule<TDataType>>();
		m_horizon.connect(m_plasticity->inHorizon());
		this->varTimeStep()->connect(m_plasticity->inTimeStep());
		this->statePosition()->connect(m_plasticity->inPosition());
		this->stateVelocity()->connect(m_plasticity->inVelocity());
		this->stateRestShape()->connect(m_plasticity->inRestShape());
		m_nbrQuery->outNeighborIds()->connect(m_plasticity->inNeighborIds());
		this->animationPipeline()->pushModule(m_plasticity);

		m_visModule = std::make_shared<ImplicitViscosity<TDataType>>();
		m_visModule->varViscosity()->setValue(Real(1));
		this->varTimeStep()->connect(m_visModule->inTimeStep());
		m_horizon.connect(m_visModule->inSmoothingLength());
		this->statePosition()->connect(m_visModule->inPosition());
		this->stateVelocity()->connect(m_visModule->inVelocity());
		m_nbrQuery->outNeighborIds()->connect(m_visModule->inNeighborIds());
		this->animationPipeline()->pushModule(m_visModule);
	}

	template<typename TDataType>
	ElastoplasticBody<TDataType>::~ElastoplasticBody()
	{
		
	}

	template<typename TDataType>
	void ElastoplasticBody<TDataType>::resetStates()
	{
		ParticleSystem<TDataType>::resetStates();

		auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		m_horizon.connect(nbrQuery->inRadius());
		this->statePosition()->connect(nbrQuery->inPosition());
		nbrQuery->update();

		if (!this->statePosition()->isEmpty())
		{
			this->stateRestShape()->allocate();
			auto nbrPtr = this->stateRestShape()->getDataPtr();
			nbrPtr->resize(nbrQuery->outNeighborIds()->getData());

			constructRestShape(*nbrPtr, nbrQuery->outNeighborIds()->getData(), this->statePosition()->getData());
		}
	}

	template<typename TDataType>
	void ElastoplasticBody<TDataType>::updateTopology()
	{
		auto ptSet = TypeInfo::cast<PointSet<TDataType>>(this->stateTopology()->getDataPtr());
		auto& pts = ptSet->getPoints();
		pts.assign(this->statePosition()->getData());

		auto tMappings = this->getTopologyMappingList();
		for (auto iter = tMappings.begin(); iter != tMappings.end(); iter++)
		{
			(*iter)->apply();
		}
	}

	DEFINE_CLASS(ElastoplasticBody);
}