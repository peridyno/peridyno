#include "ElasticBody.h"
#include "Topology/TriangleSet.h"
#include "Topology/PointSet.h"
#include "Mapping/PointSetToPointSet.h"

#include "Collision/NeighborPointQuery.h"

#include "Module/Peridynamics.h"

#include "SharedFunc.h"

namespace dyno
{
	IMPLEMENT_TCLASS(ElasticBody, TDataType)

	template<typename TDataType>
	ElasticBody<TDataType>::ElasticBody()
		: ParticleSystem<TDataType>()
	{
		this->varHorizon()->setValue(0.0085);

		auto peri = std::make_shared<Peridynamics<TDataType>>();
		this->stateTimeStep()->connect(peri->inTimeStep());
		this->stateReferencePosition()->connect(peri->inX());
		this->statePosition()->connect(peri->inY());
		this->stateVelocity()->connect(peri->inVelocity());
		this->stateForce()->connect(peri->inForce());
		this->stateBonds()->connect(peri->inBonds());
		this->animationPipeline()->pushModule(peri);
	}

	template<typename TDataType>
	ElasticBody<TDataType>::~ElasticBody()
	{
		
	}

	template<typename TDataType>
	void ElasticBody<TDataType>::loadParticles(std::string filename)
	{
		this->statePointSet()->getDataPtr()->loadObjFile(filename);
	}

	template<typename TDataType>
	void ElasticBody<TDataType>::loadParticles(Coord lo, Coord hi, Real distance)
	{
		std::vector<Coord> vertList;
		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p = Coord(x, y, z);
					vertList.push_back(Coord(x, y, z));
				}
			}
		}

		auto ptSet = this->statePointSet()->getDataPtr();
		ptSet->setPoints(vertList);

		std::cout << "particle number: " << vertList.size() << std::endl;

		vertList.clear();
	}

	template<typename TDataType>
	void ElasticBody<TDataType>::updateTopology()
	{
		auto ptSet = this->statePointSet()->getDataPtr();
		auto& pts = ptSet->getPoints();
		pts.assign(this->statePosition()->getData());
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
			this->stateBonds()->allocate();
			auto nbrPtr = this->stateBonds()->getDataPtr();
			nbrPtr->resize(nbrQuery->outNeighborIds()->getData());

			constructRestShape(*nbrPtr, nbrQuery->outNeighborIds()->getData(), this->statePosition()->getData());

			this->stateReferencePosition()->assign(this->statePosition()->getData());
		}
	}

	DEFINE_CLASS(ElasticBody);
}