#include "CapillaryWave.h"
#include "Topology/TriangleSet.h"
#include "Topology/PointSet.h"
#include "Mapping/PointSetToPointSet.h"

#include "ParticleSystem/ParticleIntegrator.h"

#include "Topology/NeighborPointQuery.h"

#include "CapillaryWaveModule.h"
#include "Peridynamics/Peridynamics.h"
#include "Peridynamics/FixedPoints.h"

#include "SharedFunc.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(CapillaryWave, TDataType)

	template<typename TDataType>
	CapillaryWave<TDataType>::CapillaryWave(std::string name)
		: Node()
	{
		auto ptSet = std::make_shared<PointSet<TDataType>>();
		this->currentTopology()->setDataPtr(ptSet);
		/*
		auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->currentPosition()->connect(integrator->inPosition());
		this->currentVelocity()->connect(integrator->inVelocity());
		this->currentForce()->connect(integrator->inForceDensity());

		this->animationPipeline()->pushModule(integrator);

		
		auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		this->varHorizon()->connect(nbrQuery->inRadius());
		this->currentPosition()->connect(nbrQuery->inPosition());
		this->animationPipeline()->pushModule(nbrQuery);
		
		auto elasticity = std::make_shared<CapillaryWaveModule<TDataType>>();
		this->varHorizon()->connect(elasticity->inHorizon());
		this->varTimeStep()->connect(elasticity->inTimeStep());
		this->currentPosition()->connect(elasticity->inPosition());
		this->currentVelocity()->connect(elasticity->inVelocity());
		this->currentRestShape()->connect(elasticity->inRestShape());
		//nbrQuery->outNeighborIds()->connect(elasticity->inNeighborIds());
		this->animationPipeline()->pushModule(elasticity);
	*/
		
		auto fixed = std::make_shared<FixedPoints<TDataType>>();

		//Create a node for surface mesh rendering
		mSurfaceNode = this->template createAncestor<Node>("Mesh");

		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		this->currentTopology()->setDataPtr(triSet);

		mSurfaceNode->currentTopology()->setDataPtr(triSet);
		
	
	}

	template<typename TDataType>
	CapillaryWave<TDataType>::~CapillaryWave()
	{
		
	}


	template<typename TDataType>
	void CapillaryWave<TDataType>::updateTopology()
	{
		//auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->currentTopology()->getDataPtr());
		//this->currentTopology()->setDataPtr(triSet);
		//triSet->getPoints().assign(this->currentPosition()->getData());

		if (!this->currentPosition()->isEmpty())
		{
			auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->currentTopology()->getDataPtr());
			int num = this->currentPosition()->getElementCount();
			auto& pts = triSet->getPoints();
			if (num != pts.size())
			{
				pts.resize(num);
			}

			triSet->getPoints().assign(this->currentPosition()->getData());
		}
	}


	template<typename TDataType>
	void CapillaryWave<TDataType>::resetStates()
	{

		/*
		ParticleSystem<TDataType>::resetStates();

		auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		this->varHorizon()->connect(nbrQuery->inRadius());
		this->currentPosition()->connect(nbrQuery->inPosition());
		nbrQuery->update();

		if (!this->currentPosition()->isEmpty())
		{
			this->currentRestShape()->allocate();
			auto nbrPtr = this->currentRestShape()->getDataPtr();
			nbrPtr->resize(nbrQuery->outNeighborIds()->getData());

			constructRestShape(*nbrPtr, nbrQuery->outNeighborIds()->getData(), this->currentPosition()->getData());
		}
		*/
	}

	template<typename TDataType>
	void CapillaryWave<TDataType>::loadParticles(std::string filename)
	{
		auto ptSet = TypeInfo::cast<PointSet<TDataType>>(this->currentTopology()->getDataPtr());
		ptSet->loadObjFile(filename);
	}

	template<typename TDataType>
	void CapillaryWave<TDataType>::loadSurface(std::string filename)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(mSurfaceNode->currentTopology()->getDataPtr())->loadObjFile(filename);
	}

	template<typename TDataType>
	std::shared_ptr<Node> CapillaryWave<TDataType>::getSurface()
	{
		return mSurfaceNode;
	}

	DEFINE_CLASS(CapillaryWave);
}