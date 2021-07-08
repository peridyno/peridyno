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
// 		auto peri = std::make_shared<Peridynamics<TDataType>>();
// 		this->setNumericalModel(peri);
// 		this->currentPosition()->connect(&peri->m_position);
// 		this->currentVelocity()->connect(&peri->m_velocity);
// 		this->currentForce()->connect(&peri->m_forceDensity);
		auto m_integrator = this->template setNumericalIntegrator<ParticleIntegrator<TDataType>>("integrator");
		this->currentPosition()->connect(m_integrator->inPosition());
		this->currentVelocity()->connect(m_integrator->inVelocity());
		this->currentForce()->connect(m_integrator->inForceDensity());

		this->animationPipeline()->push_back(m_integrator);

		auto m_nbrQuery = this->template addComputeModule<NeighborPointQuery<TDataType>>("neighborhood");
		this->varHorizon()->connect(m_nbrQuery->inRadius());
		this->currentPosition()->connect(m_nbrQuery->inPosition());

		this->animationPipeline()->push_back(m_nbrQuery);


		auto m_elasticity = this->template addConstraintModule<ElasticityModule<TDataType>>("elasticity");
		this->varHorizon()->connect(m_elasticity->inHorizon());
		this->currentPosition()->connect(m_elasticity->inPosition());
		this->currentVelocity()->connect(m_elasticity->inVelocity());
		this->currentRestShape()->connect(m_elasticity->inRestShape());
		m_nbrQuery->outNeighborIds()->connect(m_elasticity->inNeighborIds());

		this->animationPipeline()->push_back(m_elasticity);


		auto fixed = std::make_shared<FixedPoints<TDataType>>();

		//Create a node for surface mesh rendering
		mSurfaceNode = this->template createAncestor<Node>("Mesh");

		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		mSurfaceNode->setTopologyModule(triSet);

// 		std::shared_ptr<PointSetToPointSet<TDataType>> surfaceMapping = std::make_shared<PointSetToPointSet<TDataType>>(this->m_pSet, triSet);
// 		this->addTopologyMapping(surfaceMapping);
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
	bool Cloth<TDataType>::initialize()
	{
		return ParticleSystem<TDataType>::initialize();
	}

	template<typename TDataType>
	void Cloth<TDataType>::advance(Real dt)
	{
// 		auto nModel = this->getNumericalModel();
// 		nModel->step(this->getDt());

		auto integrator = this->template getModule<ParticleIntegrator<TDataType>>("integrator");

		auto module = this->template getModule<ElasticityModule<TDataType>>("elasticity");

		integrator->begin();

		integrator->integrate();

		if (module != nullptr && self_update)
			module->update();

		integrator->end();
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
	bool Cloth<TDataType>::resetStatus()
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