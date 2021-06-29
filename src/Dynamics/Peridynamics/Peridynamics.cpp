#include "Peridynamics.h"
#include "Framework/DeviceContext.h"
#include "Framework/Node.h"
#include "Topology/PointSet.h"
#include "Mapping/PointSetToPointSet.h"
#include "ParticleSystem/ParticleIntegrator.h"
#include "Topology/NeighborPointQuery.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(Peridynamics, TDataType)

	template<typename TDataType>
	Peridynamics<TDataType>::Peridynamics()
		: NumericalModel()
	{
		m_horizon.setValue(0.0085);
	}

	template<typename TDataType>
	bool Peridynamics<TDataType>::initializeImpl()
	{
		if (!isAllFieldsReady())
		{
			std::cout << "Exception: " << std::string("Peridynamics's fields are not fully initialized!") << "\n";
			return false;
		}

		m_integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		m_position.connect(m_integrator->inPosition());
		m_velocity.connect(m_integrator->inVelocity());
		m_forceDensity.connect(m_integrator->inForceDensity());
		m_integrator->initialize();

		m_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		m_horizon.connect(m_nbrQuery->inRadius());
		m_position.connect(m_nbrQuery->inPosition());
		m_nbrQuery->initialize();
		m_nbrQuery->compute();

		m_elasticity = std::make_shared<ElasticityModule<TDataType>>();
		m_position.connect(m_elasticity->inPosition());
		m_velocity.connect(m_elasticity->inVelocity());
		m_horizon.connect(m_elasticity->inHorizon());
		m_nbrQuery->outNeighborIds()->connect(m_elasticity->inNeighborIds());
		m_elasticity->initialize();

		m_nbrQuery->setParent(getParent());
		m_integrator->setParent(getParent());
		m_elasticity->setParent(getParent());

		return true;
	}

	template<typename TDataType>
	void Peridynamics<TDataType>::step(Real dt)
	{
		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Parent not set for ParticleSystem!");
			return;
		}

		m_integrator->begin();

		m_integrator->integrate();

		m_elasticity->constrain();

		m_integrator->end();
	}

	DEFINE_CLASS(Peridynamics);
}