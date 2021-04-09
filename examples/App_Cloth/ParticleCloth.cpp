#include "ParticleCloth.h"
#include "Topology/TriangleSet.h"
#include "Topology/PointSet.h"
#include "Mapping/PointSetToPointSet.h"
#include "SurfaceMeshRender.h"
#include "PointRenderModule.h"
#include "ParticleSystem/ElasticityModule.h"
#include "ParticleSystem/Peridynamics.h"
#include "ParticleSystem/FixedPoints.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(ParticleCloth, TDataType)

	template<typename TDataType>
	ParticleCloth<TDataType>::ParticleCloth(std::string name)
		: ParticleSystem<TDataType>(name)
	{
		auto peri = std::make_shared<Peridynamics<TDataType>>();
		this->setNumericalModel(peri);
		this->currentPosition()->connect(&peri->m_position);
		this->currentVelocity()->connect(&peri->m_velocity);
		this->currentForce()->connect(&peri->m_forceDensity);

		auto fixed = std::make_shared<FixedPoints<TDataType>>();


		//Create a node for surface mesh rendering
		m_surfaceNode = this->template createChild<Node>("Mesh");

		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		m_surfaceNode->setTopologyModule(triSet);

		auto render = std::make_shared<SurfaceMeshRender>();
		render->setColor(Vector3f(0.4, 0.75, 1));
		m_surfaceNode->addVisualModule(render);

		std::shared_ptr<PointSetToPointSet<TDataType>> surfaceMapping = std::make_shared<PointSetToPointSet<TDataType>>(this->m_pSet, triSet);
		this->addTopologyMapping(surfaceMapping);

		this->setVisible(true);
	}

	template<typename TDataType>
	ParticleCloth<TDataType>::~ParticleCloth()
	{
		
	}

	template<typename TDataType>
	bool ParticleCloth<TDataType>::translate(Coord t)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->translate(t);

		return ParticleSystem<TDataType>::translate(t);
	}


	template<typename TDataType>
	bool ParticleCloth<TDataType>::scale(Real s)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->scale(s);

		return ParticleSystem<TDataType>::scale(s);
	}


	template<typename TDataType>
	bool ParticleCloth<TDataType>::initialize()
	{
		return ParticleSystem<TDataType>::initialize();
	}


	template<typename TDataType>
	void ParticleCloth<TDataType>::advance(Real dt)
	{
		auto nModel = this->getNumericalModel();
		nModel->step(this->getDt());
	}

	template<typename TDataType>
	void ParticleCloth<TDataType>::updateTopology()
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
	void ParticleCloth<TDataType>::loadSurface(std::string filename)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->loadObjFile(filename);
	}

	DEFINE_CLASS(ParticleCloth);
}