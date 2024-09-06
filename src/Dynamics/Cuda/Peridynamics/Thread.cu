#include "Thread.h"
#include "Topology/TriangleSet.h"
#include "Topology/PointSet.h"
#include "Mapping/PointSetToPointSet.h"

#include "ParticleSystem/Module/ParticleIntegrator.h"

#include "Collision/NeighborPointQuery.h"

#include "Module/LinearElasticitySolver.h"
#include "Module/Peridynamics.h"
#include "Module/FixedPoints.h"

#include "SharedFunc.h"
#include "TriangularSystem.h"

namespace dyno
{
	IMPLEMENT_TCLASS(Thread, TDataType)

	template<typename TDataType>
	Thread<TDataType>::Thread()
		: ThreadSystem<TDataType>()
	{
		auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->stateTimeStep()->connect(integrator->inTimeStep());
		this->statePosition()->connect(integrator->inPosition());
		this->stateVelocity()->connect(integrator->inVelocity());
		this->stateForce()->connect(integrator->inForceDensity());

		this->animationPipeline()->pushModule(integrator);

		auto elasticity = std::make_shared<LinearElasticitySolver<TDataType>>();
		this->varHorizon()->connect(elasticity->inHorizon());
		this->stateTimeStep()->connect(elasticity->inTimeStep());
		this->statePosition()->connect(elasticity->inY());
		this->stateVelocity()->connect(elasticity->inVelocity());
		this->stateRestShape()->connect(elasticity->inBonds());
		this->animationPipeline()->pushModule(elasticity);

		
	}

	template<typename TDataType>
	Thread<TDataType>::~Thread()
	{
		
	}

	template<typename TDataType>
	bool Thread<TDataType>::translate(Coord t)
	{
		this->stateEdgeSet()->getDataPtr()->translate(t);

		return true;
	}

	template<typename TDataType>
	bool Thread<TDataType>::scale(Real s)
	{
		this->stateEdgeSet()->getDataPtr()->scale(s);
		return true;
	}

	


	template<typename TDataType>
	void Thread<TDataType>::resetStates()
	{
		ThreadSystem<TDataType>::resetStates();

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
			//constructRestShapeForThreads();
		}
	}

	
	

	DEFINE_CLASS(Thread);
}