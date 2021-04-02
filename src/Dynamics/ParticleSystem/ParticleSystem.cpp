#include "ParticleSystem.h"
#include "PositionBasedFluidModel.h"

#include "Topology/PointSet.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(ParticleSystem, TDataType)

	template<typename TDataType>
	ParticleSystem<TDataType>::ParticleSystem(std::string name)
		: Node(name)
	{
//		attachField(&m_velocity, MechanicalState::velocity(), "Storing the particle velocities!", false);
//		attachField(&m_force, MechanicalState::force(), "Storing the force densities!", false);

		m_pSet = std::make_shared<PointSet<TDataType>>();
		this->setTopologyModule(m_pSet);

// 		m_pointsRender = std::make_shared<PointRenderModule>();
// 		this->addVisualModule(m_pointsRender);
	}

	template<typename TDataType>
	ParticleSystem<TDataType>::~ParticleSystem()
	{
		
	}


	template<typename TDataType>
	void ParticleSystem<TDataType>::loadParticles(std::string filename)
	{
		m_pSet->loadObjFile(filename);
	}

	template<typename TDataType>
	void ParticleSystem<TDataType>::loadParticles(Coord center, Real r, Real distance)
	{
		std::vector<Coord> vertList;

		Coord lo = center - r;
		Coord hi = center + r;

		for (Real x = lo[0]; x <= hi[0]; x +=  distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p = Coord(x, y, z);
					if ((p-center).norm() < r)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		m_pSet->setPoints(vertList);

		vertList.clear();
	}

	template<typename TDataType>
	void ParticleSystem<TDataType>::loadParticles(Coord lo, Coord hi, Real distance)
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

		m_pSet->setPoints(vertList);

		std::cout << "particle number: " << vertList.size() << std::endl;

		vertList.clear();
	}

	template<typename TDataType>
	bool ParticleSystem<TDataType>::translate(Coord t)
	{
		m_pSet->translate(t);

		return true;
	}


	template<typename TDataType>
	bool ParticleSystem<TDataType>::scale(Real s)
	{
		m_pSet->scale(s);

		return true;
	}

	template<typename TDataType>
	bool ParticleSystem<TDataType>::initialize()
	{
		return Node::initialize();
	}

// 	template<typename TDataType>
// 	void ParticleSystem<TDataType>::setVisible(bool visible)
// 	{
// 		if (m_pointsRender == nullptr)
// 		{
// 			m_pointsRender = std::make_shared<PointRenderModule>();
// 			this->addVisualModule(m_pointsRender);
// 		}
// 
// 		Node::setVisible(visible);
// 	}

	template<typename TDataType>
	void ParticleSystem<TDataType>::updateTopology()
	{
		if (!this->currentPosition()->isEmpty())
		{
			int num = this->currentPosition()->getElementCount();
			auto& pts = m_pSet->getPoints();
			if (num != pts.size())
			{
				pts.resize(num);
			}

			pts.assign(this->currentPosition()->getValue());
		}
	}


	template<typename TDataType>
	bool ParticleSystem<TDataType>::resetStatus()
	{
		auto ptSet = TypeInfo::cast<PointSet<TDataType>>(this->getTopologyModule());
		if (ptSet == nullptr) return false;

		auto pts = ptSet->getPoints();

		if (pts.size() > 0)
		{
			this->currentPosition()->setElementCount(pts.size());
			this->currentVelocity()->setElementCount(pts.size());
			this->currentForce()->setElementCount(pts.size());

			this->currentPosition()->getValue().assign(pts);
			this->currentVelocity()->getReference()->reset();
		}

		return Node::resetStatus();
	}

	DEFINE_CLASS(ParticleSystem);
}