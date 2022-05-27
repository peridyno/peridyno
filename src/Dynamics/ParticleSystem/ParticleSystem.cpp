#include "ParticleSystem.h"
#include "Topology/PointSet.h"

namespace dyno
{
	IMPLEMENT_TCLASS(ParticleSystem, TDataType)

	template<typename TDataType>
	ParticleSystem<TDataType>::ParticleSystem(std::string name)
		: Node(name)
	{
		auto ptSet = std::make_shared<PointSet<TDataType>>();
		this->stateTopology()->setDataPtr(ptSet);
	}

	template<typename TDataType>
	ParticleSystem<TDataType>::~ParticleSystem()
	{
	}

	template<typename TDataType>
	std::string ParticleSystem<TDataType>::getNodeType()
	{
		return "ParticleSystem";
	}

	template<typename TDataType>
	void ParticleSystem<TDataType>::loadParticles(std::string filename)
	{
		auto ptSet = TypeInfo::cast<PointSet<TDataType>>(this->stateTopology()->getDataPtr());
		ptSet->loadObjFile(filename);
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

		auto ptSet = TypeInfo::cast<PointSet<TDataType>>(this->stateTopology()->getDataPtr());
		ptSet->setPoints(vertList);

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

		auto ptSet = TypeInfo::cast<PointSet<TDataType>>(this->stateTopology()->getDataPtr());
		ptSet->setPoints(vertList);

		std::cout << "particle number: " << vertList.size() << std::endl;

		vertList.clear();
	}

	template<typename TDataType>
	bool ParticleSystem<TDataType>::translate(Coord t)
	{
		auto ptSet = TypeInfo::cast<PointSet<TDataType>>(this->stateTopology()->getDataPtr());
		ptSet->translate(t);

		return true;
	}


	template<typename TDataType>
	bool ParticleSystem<TDataType>::scale(Real s)
	{
		auto ptSet = TypeInfo::cast<PointSet<TDataType>>(this->stateTopology()->getDataPtr());
		ptSet->scale(s);

		return true;
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
		if (!this->statePosition()->isEmpty())
		{
			auto ptSet = TypeInfo::cast<PointSet<TDataType>>(this->stateTopology()->getDataPtr());
			int num = this->statePosition()->getElementCount();
			auto& pts = ptSet->getPoints();
			if (num != pts.size())
			{
				pts.resize(num);
			}

			pts.assign(this->statePosition()->getData());
		}
	}


	template<typename TDataType>
	void ParticleSystem<TDataType>::resetStates()
	{
		auto ptSet = TypeInfo::cast<PointSet<TDataType>>(this->stateTopology()->getDataPtr());
		if (ptSet == nullptr) return;

		auto pts = ptSet->getPoints();

		if (pts.size() > 0)
		{
			this->statePosition()->setElementCount(pts.size());
			this->stateVelocity()->setElementCount(pts.size());
			this->stateForce()->setElementCount(pts.size());

			this->statePosition()->getData().assign(pts);
			this->stateVelocity()->getDataPtr()->reset();
		}

		Node::resetStates();
	}

	DEFINE_CLASS(ParticleSystem);
}