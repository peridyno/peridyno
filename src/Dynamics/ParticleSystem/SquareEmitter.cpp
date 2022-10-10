#include "SquareEmitter.h"
#include <time.h>

#include <stdlib.h>

namespace dyno
{
	template<typename TDataType>
	SquareEmitter<TDataType>::SquareEmitter(std::string name)
		: ParticleEmitter<TDataType>(name)
	{
		srand(time(0));

		this->varSamplingDistance()->setRange(0.001, 1.0);
		this->varVelocityMagnitude()->setRange(0.0, 10.0);

		this->varWidth()->setRange(0.01, 10.0f);
		this->varHeight()->setRange(0.01, 10.0f);

		this->stateOutline()->setDataPtr(std::make_shared<EdgeSet<TDataType>>());
	}

	template<typename TDataType>
	SquareEmitter<TDataType>::~SquareEmitter()
	{
	}

	template<typename TDataType>
	void SquareEmitter<TDataType>::generateParticles()
	{
		auto sampling_distance = this->varSamplingDistance()->getData();
		if (sampling_distance < EPSILON)
			sampling_distance = Real(0.005);
		auto center = this->varLocation()->getData();

		std::vector<Coord> pos_list;
		std::vector<Coord> vel_list;

		auto rot_vec = this->varRotation()->getData();

		auto rot_mat = this->rotationMatrix();

		Coord v0 = this->varVelocityMagnitude()->getData()*rot_mat*Vec3f(0, -1, 0);

		auto w = 0.5 * this->varWidth()->getData();
		auto h = 0.5 * this->varHeight()->getData();

		for (Real x = -w; x <= w; x += sampling_distance)
		{
			for (Real z = -h; z <= h; z += sampling_distance)
			{
				Coord p = Coord(x, 0, z);
				if (rand() % 40 == 0)
				{
					//Coord q = cos(angle) * p + (1 - cos(angle)) * (p.dot(axis)) * axis + sin(angle) * axis.cross(p);
					Coord q = rot_mat * p;
					pos_list.push_back(q + center);
					vel_list.push_back(v0);
				}
			}
		}

		if (pos_list.size() > 0)
		{
			this->mPosition.resize(pos_list.size());
			this->mVelocity.resize(pos_list.size());

			this->mPosition.assign(pos_list);
			this->mVelocity.assign(vel_list);
		}


		pos_list.clear();
		vel_list.clear();
	}
	
	template<typename TDataType>
	void SquareEmitter<TDataType>::resetStates()
	{
		std::vector<Coord> vertices;
		std::vector<TopologyModule::Edge> edges;

		auto center = this->varLocation()->getData();
		auto rot_vec = this->varRotation()->getData();

		auto w = this->varWidth()->getData();
		auto h = this->varHeight()->getData();

		auto rot_mat = this->rotationMatrix();

		auto Nx = 0.5 * w * rot_mat * Coord(1, 0, 0);
		auto Nz = 0.5 * h * rot_mat * Coord(0, 0, 1);

		vertices.push_back(center + Nx + Nz);
		vertices.push_back(center + Nx - Nz);
		vertices.push_back(center - Nx - Nz);
		vertices.push_back(center - Nx + Nz);

		edges.push_back(TopologyModule::Edge(0, 1));
		edges.push_back(TopologyModule::Edge(1, 2));
		edges.push_back(TopologyModule::Edge(2, 3));
		edges.push_back(TopologyModule::Edge(3, 0));

		auto edgeTopo = this->stateOutline()->getDataPtr();

		edgeTopo->setPoints(vertices);
		edgeTopo->setEdges(edges);

		vertices.clear();
		edges.clear();
	}

	DEFINE_CLASS(SquareEmitter);
}