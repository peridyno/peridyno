#include "Cloth.h"
#include "Topology/TriangleSet.h"
#include "Topology/PointSet.h"
#include "Mapping/PointSetToPointSet.h"

#include "Peridynamics/ElasticityModule.h"
#include "Peridynamics/Peridynamics.h"
#include "Peridynamics/FixedPoints.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(Cloth, TDataType)

	template<typename TDataType>
	Cloth<TDataType>::Cloth(std::string name)
		: ParticleSystem<TDataType>(name)
	{
		auto peri = std::make_shared<Peridynamics<TDataType>>();
		this->setNumericalModel(peri);
		this->currentPosition()->connect(&peri->m_position);
		this->currentVelocity()->connect(&peri->m_velocity);
		this->currentForce()->connect(&peri->m_forceDensity);

		auto fixed = std::make_shared<FixedPoints<TDataType>>();

		//Create a node for surface mesh rendering
		mSurfaceNode = this->template createChild<Node>("Mesh");

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
		auto nModel = this->getNumericalModel();
		nModel->step(this->getDt());
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