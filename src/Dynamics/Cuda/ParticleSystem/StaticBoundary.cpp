#include "StaticBoundary.h"
#include "Log.h"
#include "Node.h"

#include "Module/BoundaryConstraint.h"

#include "Topology/DistanceField3D.h"
#include "Topology/TriangleSet.h"

namespace dyno
{
	IMPLEMENT_TCLASS(StaticBoundary, TDataType)

	template<typename TDataType>
	StaticBoundary<TDataType>::StaticBoundary()
		: Node()
	{
		this->varNormalFriction()->setValue(0.95);
		this->varTangentialFriction()->setValue(0.0);
	}

	template<typename TDataType>
	StaticBoundary<TDataType>::~StaticBoundary()
	{
	}

// 	template<typename TDataType>
// 	bool StaticBoundary<TDataType>::addRigidBody(std::shared_ptr<RigidBody<TDataType>> child)
// 	{
// 		this->addChild(child);
// 		m_rigids.push_back(child);
// 		return true;
// 	}

// 	template<typename TDataType>
// 	bool StaticBoundary<TDataType>::addParticleSystem(std::shared_ptr<ParticleSystem<TDataType>> child)
// 	{
// 		this->addChild(child);
// 		m_particleSystems.push_back(child);
// 
// 		this->inportParticleSystems()->addNode(child);
// 
// 		return true;
// 	}

	template<typename TDataType>
	void StaticBoundary<TDataType>::updateStates()
	{
		Real dt = this->stateTimeStep()->getData();

		auto pSys = this->getParticleSystems();

		for (size_t t = 0; t < m_obstacles.size(); t++)
		{

			for (int i = 0; i < pSys.size(); i++)
			{
				DeviceArrayField<Coord>* posFd = pSys[i]->statePosition();
				DeviceArrayField<Coord>* velFd = pSys[i]->stateVelocity();
				m_obstacles[t]->constrain(posFd->getData(), velFd->getData(), dt);
			}
		} 
	}

	template<typename TDataType>
	void StaticBoundary<TDataType>::loadSDF(std::string filename, bool bOutBoundary)
	{
		auto boundary = std::make_shared<BoundaryConstraint<TDataType>>();
		boundary->load(filename, bOutBoundary);

		this->varNormalFriction()->connect(boundary->varNormalFriction());
		this->varTangentialFriction()->connect(boundary->varTangentialFriction());

		m_obstacles.push_back(boundary);
	}


	template<typename TDataType>
	std::shared_ptr<Node> StaticBoundary<TDataType>::loadCube(Coord lo, Coord hi, Real distance, bool bOutBoundary /*= false*/)
	{
		auto boundary = std::make_shared<BoundaryConstraint<TDataType>>();
		boundary->setCube(lo, hi, distance, bOutBoundary);

		this->varNormalFriction()->connect(boundary->varNormalFriction());
		this->varTangentialFriction()->connect(boundary->varTangentialFriction());

		m_obstacles.push_back(boundary);

		//Note: the size of standard cube is 2m*2m*2m
		Coord scale = (hi - lo) / 2;
		Coord center = (hi + lo) / 2;

		return nullptr;
	}

	template<typename TDataType>
	void StaticBoundary<TDataType>::loadShpere(Coord center, Real r, Real distance, bool bOutBoundary /*= false*/, bool bVisible)
	{
		auto boundary = std::make_shared<BoundaryConstraint<TDataType>>();
		boundary->setSphere(center, r, distance, bOutBoundary);

		this->varNormalFriction()->connect(boundary->varNormalFriction());
		this->varTangentialFriction()->connect(boundary->varTangentialFriction());

		m_obstacles.push_back(boundary);
	}


	template<typename TDataType>
	void StaticBoundary<TDataType>::scale(Real s)
	{
		for (int i = 0; i < m_obstacles.size(); i++)
		{
			m_obstacles[i]->m_cSDF->scale(s);
		}
	}

	template<typename TDataType>
	void StaticBoundary<TDataType>::translate(Coord t)
	{
		for (int i = 0; i < m_obstacles.size(); i++)
		{
			m_obstacles[i]->m_cSDF->translate(t);
		}
	}



	template<typename TDataType>
	void StaticBoundary<TDataType>::resetStates()
	{
		auto filename = this->varFileName()->getData();
	}

	DEFINE_CLASS(StaticBoundary);
}