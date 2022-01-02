#include "GhostParticles.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(GhostParticles, TDataType)

	template<typename TDataType>
	GhostParticles<TDataType>::GhostParticles()
		: ParticleSystem<TDataType>()
	{
	}

	template<typename TDataType>
	GhostParticles<TDataType>::~GhostParticles()
	{
	}

	template<typename TDataType>
	void GhostParticles<TDataType>::loadPlane()
	{
		std::vector<Coord> host_pos;
		std::vector<Coord> host_vel;
		std::vector<Coord> host_force;
		std::vector<Coord> host_normal;
		std::vector<Attribute> host_attribute;

		Coord low(-0.2, -0.015, -0.2);
		Coord high(0.2, -0.005, 0.2);

		Real s = 0.005f;
		int m_iExt = 0;

		float omega = 1.0f;
		float half_s = -s / 2.0f;

		int num = 0;

		for (float x = low.x - m_iExt * s; x <= high.x + m_iExt * s; x += s) {
			for (float y = low.y - m_iExt * s; y <= high.y + m_iExt * s; y += s) {
				for (float z = low.z - m_iExt * s; z <= high.z + m_iExt * s; z += s) {
					Attribute attri;
					attri.setFluid();
					attri.setDynamic();

					host_pos.push_back(Coord(x, y, z));
					host_vel.push_back(Coord(0));
					host_force.push_back(Coord(0));
					host_normal.push_back(Coord(0, 1, 0));
					host_attribute.push_back(attri);
				}
			}
		}

		this->currentPosition()->setElementCount(num);
		this->currentVelocity()->setElementCount(num);
		this->currentForce()->setElementCount(num);

		this->stateNormal()->setElementCount(num);
		this->stateAttribute()->setElementCount(num);

		this->currentPosition()->getDataPtr()->assign(host_pos);
		this->currentVelocity()->getDataPtr()->assign(host_vel);
		this->currentForce()->getDataPtr()->assign(host_force);
		this->stateNormal()->getDataPtr()->assign(host_normal);
		this->stateAttribute()->getDataPtr()->assign(host_attribute);

		this->updateTopology();
	}

// 	template<typename TDataType>
// 	void GhostParticles<TDataType>::updateTopology()
// 	{
// 		if (!this->currentPosition()->isEmpty())
// 		{
// 			auto ptSet = TypeInfo::cast<PointSet<TDataType>>(this->currentTopology()->getDataPtr());
// 			int num = this->currentPosition()->getElementCount();
// 			auto& pts = ptSet->getPoints();
// 			if (num != pts.size())
// 			{
// 				pts.resize(num);
// 			}
// 
// 			pts.assign(this->currentPosition()->getData());
// 		}
// 	}

	DEFINE_CLASS(GhostParticles);
}