#include "ParticleViscoplasticBody.h"
#include "Topology/TriangleSet.h"
#include "Topology/PointSet.h"
#include "SurfaceMeshRender.h"
#include "PointRenderModule.h"
#include "Mapping/PointSetToPointSet.h"
#include "Topology/NeighborPointQuery.h"
#include "ParticleSystem/ParticleIntegrator.h"

#include "Peridynamics/ElastoplasticityModule.h"

#include "ParticleSystem/DensityPBD.h"
#include "ParticleSystem/ImplicitViscosity.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(ParticleViscoplasticBody, TDataType)

	template<typename TDataType>
	ParticleViscoplasticBody<TDataType>::ParticleViscoplasticBody(std::string name)
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

		m_plasticity = this->template addConstraintModule<ElastoplasticityModule<TDataType>>("elastopolasticity");
		this->currentPosition()->connect(m_plasticity->inPosition());
		this->currentVelocity()->connect(m_plasticity->inVelocity());
		m_nbrQuery->outNeighborIds()->connect(m_plasticity->inNeighborIds());
		m_plasticity->setFrictionAngle(0);
		m_plasticity->setCohesion(0.0);
		m_plasticity->enableFullyReconstruction();

// 		m_pbdModule = this->template addConstraintModule<DensityPBD<TDataType>>("pbd");
// 		m_horizon.connect(m_pbdModule->varSmoothingLength());
// 		this->currentPosition()->connect(m_pbdModule->inPosition());
// 		this->currentVelocity()->connect(m_pbdModule->inVelocity());
// 		m_nbrQuery->outNeighborhood()->connect(m_pbdModule->inNeighborIndex());

		m_visModule = this->template addConstraintModule<ImplicitViscosity<TDataType>>("viscosity");
		m_visModule->setViscosity(Real(1));
		m_horizon.connect(&m_visModule->m_smoothingLength);
		this->currentPosition()->connect(&m_visModule->m_position);
		this->currentVelocity()->connect(&m_visModule->m_velocity);
		m_nbrQuery->outNeighborIds()->connect(m_visModule->inNeighborIds());

		m_surfaceNode = this->template createChild<Node>("Mesh");

		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		m_surfaceNode->setTopologyModule(triSet);

		auto render = std::make_shared<SurfaceMeshRender>();
		render->setColor(Vec3f(0.2f, 0.6, 1.0f));
		m_surfaceNode->addVisualModule(render);

		m_surfaceNode->setVisible(false);

		std::shared_ptr<PointSetToPointSet<TDataType>> surfaceMapping = std::make_shared<PointSetToPointSet<TDataType>>(this->m_pSet, triSet);
		this->addTopologyMapping(surfaceMapping);
	}

	template<typename TDataType>
	ParticleViscoplasticBody<TDataType>::~ParticleViscoplasticBody()
	{
		
	}

	template<typename TDataType>
	void ParticleViscoplasticBody<TDataType>::advance(Real dt)
	{
		m_integrator->begin();

		m_integrator->integrate();

		//m_nbrQuery->compute();
		m_plasticity->solveElasticity();
		//m_nbrQuery->compute();

		m_plasticity->applyPlasticity();

		m_visModule->constrain();

		m_integrator->end();
	}

	template<typename TDataType>
	void ParticleViscoplasticBody<TDataType>::updateTopology()
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
	bool ParticleViscoplasticBody<TDataType>::initialize()
	{
		m_nbrQuery->initialize();
		m_nbrQuery->compute();

		return ParticleSystem<TDataType>::initialize();
	}

	template<typename TDataType>
	void ParticleViscoplasticBody<TDataType>::loadSurface(std::string filename)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->loadObjFile(filename);
	}

	template<typename TDataType>
	bool ParticleViscoplasticBody<TDataType>::translate(Coord t)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->translate(t);

		return ParticleSystem<TDataType>::translate(t);
	}

	template<typename TDataType>
	bool ParticleViscoplasticBody<TDataType>::scale(Real s)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->scale(s);

		return ParticleSystem<TDataType>::scale(s);
	}

	DEFINE_CLASS(ParticleViscoplasticBody);
}