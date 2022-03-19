#include "ElasticBody.h"
#include "Topology/TriangleSet.h"
#include "Topology/PointSet.h"
#include "Mapping/PointSetToPointSet.h"
#include "Topology/NeighborPointQuery.h"
#include "Peridynamics/Peridynamics.h"
#include "SharedFunc.h"

namespace dyno
{
	IMPLEMENT_TCLASS(ElasticBody, TDataType)

	template<typename TDataType>
	ElasticBody<TDataType>::ElasticBody(std::string name)
		: ParticleSystem<TDataType>(name)
	{
		this->varHorizon()->setValue(0.0085);

		auto peri = std::make_shared<Peridynamics<TDataType>>();
		this->varTimeStep()->connect(peri->inTimeStep());
		this->statePosition()->connect(peri->inPosition());
		this->stateVelocity()->connect(peri->inVelocity());
		this->stateForce()->connect(peri->inForce());
		this->stateRestShape()->connect(peri->inRestShape());
		this->animationPipeline()->pushModule(peri);

		//Create a node for surface mesh rendering
		m_surfaceNode = std::make_shared<Node>("Mesh");// this->template createAncestor<Node>("Mesh");
		m_surfaceNode->addAncestor(this);

		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		m_surfaceNode->stateTopology()->setDataPtr(triSet);

		//Set the topology mapping from PointSet to TriangleSet
		auto surfaceMapping = this->template addTopologyMapping<PointSetToPointSet<TDataType>>("surface_mapping");
		auto ptSet = TypeInfo::cast<PointSet<TDataType>>(this->stateTopology()->getDataPtr());
		surfaceMapping->setFrom(ptSet);
		surfaceMapping->setTo(triSet);
	}

	template<typename TDataType>
	ElasticBody<TDataType>::~ElasticBody()
	{
		
	}

	template<typename TDataType>
	bool ElasticBody<TDataType>::translate(Coord t)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->stateTopology()->getDataPtr())->translate(t);

		return ParticleSystem<TDataType>::translate(t);
	}

	template<typename TDataType>
	bool ElasticBody<TDataType>::scale(Real s)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->stateTopology()->getDataPtr())->scale(s);

		return ParticleSystem<TDataType>::scale(s);
	}

	template<typename TDataType>
	void ElasticBody<TDataType>::updateTopology()
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

	template<typename TDataType>
	void ElasticBody<TDataType>::resetStates()
	{
		ParticleSystem<TDataType>::resetStates();

		auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
 		this->varHorizon()->connect(nbrQuery->inRadius());
 		this->statePosition()->connect(nbrQuery->inPosition());
		nbrQuery->update();

		if (!this->statePosition()->isEmpty())
		{
			this->stateRestShape()->allocate();
			auto nbrPtr = this->stateRestShape()->getDataPtr();
			nbrPtr->resize(nbrQuery->outNeighborIds()->getData());

			constructRestShape(*nbrPtr, nbrQuery->outNeighborIds()->getData(), this->statePosition()->getData());

			this->stateReferencePosition()->allocate();
			this->stateReferencePosition()->getDataPtr()->assign(this->statePosition()->getData());

			this->stateNeighborIds()->allocate();
			this->stateNeighborIds()->getDataPtr()->assign(nbrQuery->outNeighborIds()->getData());
		}
	}

	template<typename TDataType>
	void ElasticBody<TDataType>::loadSurface(std::string filename)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->stateTopology()->getDataPtr())->loadObjFile(filename);
	}


	template<typename TDataType>
	std::shared_ptr<PointSetToPointSet<TDataType>> ElasticBody<TDataType>::getTopologyMapping()
	{
		auto mapping = this->template getModule<PointSetToPointSet<TDataType>>("surface_mapping");

		return mapping;
	}

	DEFINE_CLASS(ElasticBody);
}