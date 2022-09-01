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
		this->statePointSet()->setDataPtr(ptSet);
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
		this->statePointSet()->getDataPtr()->loadObjFile(filename);
	}

	template<typename TDataType>
	void ParticleSystem<TDataType>::loadParticles(Coord center, Real r, Coord center1, Real r1, Real distance)
	{
		std::vector<Coord> vertList;

		Coord lo = center - r;
		Coord hi = center + r;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p = Coord(x, y, z);
					if ((p - center).norm() < r)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center1 - r1;
		hi = center1 + r1;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p1 = Coord(x, y, z);
					if ((p1 - center1).norm() < r1)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		auto ptSet = this->statePointSet()->getDataPtr();
		ptSet->setPoints(vertList);

		vertList.clear();
	}

	template<typename TDataType>
	void ParticleSystem<TDataType>::loadParticles(Coord center, Real r,
		Coord center1, Real r1,
		Coord center2, Real r2,
		Coord center3, Real r3,
		Coord lo_cu, Coord hi_cu, Real distance)
	{
		std::vector<Coord> vertList;

		Coord lo = center - r;
		Coord hi = center + r;
		Coord ly = center - r*0.6f;;
		Coord hy = center + r * 0.05f;;
		

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = ly[1]; y <= hy[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p = Coord(x, y, z);
					if ((p - center).norm() < r)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center1 - r1;
		hi = center1 + r1;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p1 = Coord(x, y, z);
					if ((p1 - center1).norm() < r1)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center2 - r2;
		hi = center2 + r2;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p2 = Coord(x, y, z);
					if ((p2 - center2).norm() < r2)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center3 - r3;
		hi = center3 + r3;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p3 = Coord(x, y, z);
					if ((p3 - center3).norm() < r3)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		/*for (Real x = lo_cu[0]; x <= hi_cu[0]; x += distance)
		{
			for (Real y = lo_cu[1]; y <= hi_cu[1]; y += distance)
			{
				for (Real z = lo_cu[2]; z <= hi_cu[2]; z += distance)
				{
					Coord p_cu = Coord(x, y, z);
					vertList.push_back(Coord(x, y, z));
				}
			}
		}*/

		auto ptSet = this->statePointSet()->getDataPtr();
		ptSet->setPoints(vertList);

		vertList.clear();
	}

	template<typename TDataType>
	void ParticleSystem<TDataType>::loadParticles(Coord center, Real r,
												  Coord center1, Real r1,
												  Coord center2, Real r2,
												  Coord center3, Real r3,
												  Coord center4, Real r4,
												  Coord center5, Real r5,
												  Coord center6, Real r6,
												  Coord center7, Real r7,
												  Coord center8, Real r8,
												  Coord center9, Real r9,
												  Coord center10, Real r10,
												  Coord center11, Real r11,
												  Coord center12, Real r12,
												  Coord center13, Real r13,
												  Coord center14, Real r14,
												  Real distance)
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

		lo = center1 - r1;
		hi = center1 + r1;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p1 = Coord(x, y, z);
					if ((p1 - center1).norm() < r1)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center2 - r2;
		hi = center2 + r2;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p2 = Coord(x, y, z);
					if ((p2 - center2).norm() < r2)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center3 - r3;
		hi = center3 + r3;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p3 = Coord(x, y, z);
					if ((p3 - center3).norm() < r3)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center4 - r4;
		hi = center4 + r4;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p4 = Coord(x, y, z);
					if ((p4 - center4).norm() < r4)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center5 - r5;
		hi = center5 + r5;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p5 = Coord(x, y, z);
					if ((p5 - center5).norm() < r5)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center6 - r6;
		hi = center6 + r6;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p6 = Coord(x, y, z);
					if ((p6 - center6).norm() < r6)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center7 - r7;
		hi = center7 + r7;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p7 = Coord(x, y, z);
					if ((p7 - center7).norm() < r7)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center8 - r8;
		hi = center8 + r8;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p8 = Coord(x, y, z);
					if ((p8 - center8).norm() < r8)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center9 - r9;
		hi = center9 + r9;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p9 = Coord(x, y, z);
					if ((p9 - center9).norm() < r9)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center10 - r10;
		hi = center10 + r10;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p10 = Coord(x, y, z);
					if ((p10 - center10).norm() < r10)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center11 - r11;
		hi = center11 + r11;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p11 = Coord(x, y, z);
					if ((p11 - center11).norm() < r11)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center12 - r12;
		hi = center12 + r12;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p12 = Coord(x, y, z);
					if ((p12 - center12).norm() < r12)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center13 - r13;
		hi = center13 + r13;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p13 = Coord(x, y, z);
					if ((p13 - center13).norm() < r13)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center14 - r14;
		hi = center14 + r14;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p14 = Coord(x, y, z);
					if ((p14 - center14).norm() < r14)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		auto ptSet = this->statePointSet()->getDataPtr();
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

		auto ptSet = this->statePointSet()->getDataPtr();
		ptSet->setPoints(vertList);

		std::cout << "particle number: " << vertList.size() << std::endl;

		vertList.clear();
	}

	template<typename TDataType>
	bool ParticleSystem<TDataType>::translate(Coord t)
	{
		auto ptSet = this->statePointSet()->getDataPtr();
		ptSet->translate(t);

		return true;
	}


	template<typename TDataType>
	bool ParticleSystem<TDataType>::scale(Real s)
	{
		auto ptSet = this->statePointSet()->getDataPtr();
		ptSet->scale(s);

		return true;
	}


	template<typename TDataType>
	bool ParticleSystem<TDataType>::rotate(Quat<Real> q)
	{
		auto ptSet = this->statePointSet()->getDataPtr();
		ptSet->rotate(q);

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
			auto ptSet = this->statePointSet()->getDataPtr();
			int num = this->statePosition()->size();
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
		auto ptSet = this->statePointSet()->getDataPtr();
		if (ptSet == nullptr) return;

		auto pts = ptSet->getPoints();

		if (pts.size() > 0)
		{
			this->statePosition()->resize(pts.size());
			this->stateVelocity()->resize(pts.size());
			this->stateForce()->resize(pts.size());

			this->statePosition()->getData().assign(pts);
			this->stateVelocity()->getDataPtr()->reset();
		}

		Node::resetStates();
	}

	DEFINE_CLASS(ParticleSystem);
}