#include "CircularEmitter.h"
#include <time.h>

#include <stdlib.h>

namespace dyno
{
	template<typename TDataType>
	CircularEmitter<TDataType>::CircularEmitter(std::string name)
		: ParticleEmitter<TDataType>(name)
	{
		srand(time(0));

		this->varRadius()->setRange(0.0, 10.0);
		this->varSamplingDistance()->setRange(0.001, 1.0);

		this->stateOutline()->setDataPtr(std::make_shared<EdgeSet<TDataType>>());
	}

	template<typename TDataType>
	CircularEmitter<TDataType>::~CircularEmitter()
	{
		this->mPosition.clear();
	}
	

	template<typename TDataType>
	void CircularEmitter<TDataType>::generateParticles()
	{
		auto sampling_distance = this->varSamplingDistance()->getData();
		if (sampling_distance < EPSILON)
			sampling_distance = 0.005;

		auto center		= this->varLocation()->getData();
		auto rot_vec	= this->varRotation()->getData();
		auto r			= this->varRadius()->getData();

		std::vector<Coord> pos_list;
		std::vector<Coord> vel_list;

		auto rot_mat = this->rotationMatrix();

		Coord v0 = this->varVelocityMagnitude()->getData()*rot_mat*Vec3f(0, -1, 0);

		Real lo = -r;
		Real hi = r;

		for (Real x = lo; x <= hi; x += sampling_distance)
		{
			for (Real y = lo; y <= hi; y += sampling_distance)
			{
				Coord p = Coord(x, 0, y);
				if ((p - Coord(0)).norm() < r && rand() % 40 == 0)
				{
					//Coord q = cos(angle) * p + (1 - cos(angle)) * (p.dot(axis)) * axis + sin(angle) * axis.cross(p);
					Coord q = rot_mat * p;
					pos_list.push_back(q + center);
					vel_list.push_back(v0);
				}
			}
		}

		if (pos_list.size() > 0) {
			this->mPosition.resize(pos_list.size());
			this->mVelocity.resize(pos_list.size());

			this->mPosition.assign(pos_list);
			this->mVelocity.assign(vel_list);
		}
	
		
		pos_list.clear();
		vel_list.clear();
	}


	template<typename TDataType>
	void CircularEmitter<TDataType>::resetStates()
	{
		std::vector<Coord> vertices;
		std::vector<TopologyModule::Edge> edges;

		auto center = this->varLocation()->getData();
		auto rot_vec = this->varRotation()->getData();
		auto r = this->varRadius()->getData();

		auto rot_mat = this->rotationMatrix();

		int segNum = 36;
		Real deltaTheta = 2 * M_PI / segNum;

		for (int i = 0; i < segNum; i++)
		{
			Real x = r * sin(i * deltaTheta);
			Real z = r * cos(i * deltaTheta);

			vertices.push_back(center + rot_mat * Coord(x, 0, z));
			edges.push_back(TopologyModule::Edge(i, (i + 1) % segNum));
		}

		auto edgeTopo = this->stateOutline()->getDataPtr();

		edgeTopo->setPoints(vertices);
		edgeTopo->setEdges(edges);

		vertices.clear();
		edges.clear();
	}

	DEFINE_CLASS(CircularEmitter);
}