#include "SquareEmitter.h"
#include <time.h>

#include <stdlib.h>

namespace dyno
{
	IMPLEMENT_CLASS(SquareEmitter)

	SquareEmitter::SquareEmitter()
		: ParticleEmitter()
	{
		srand(time(0));

		this->varWidth()->setRange(0.01, 10.0f);
		this->varHeight()->setRange(0.01, 10.0f);

		this->stateOutline()->setDataPtr(std::make_shared<EdgeSet>());

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&SquareEmitter::tranformChanged, this));

		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);

		this->varWidth()->attach(callback);
		this->varHeight()->attach(callback);
	}

	SquareEmitter::~SquareEmitter()
	{
	}

	void SquareEmitter::generateParticles()
	{
		auto sampling_distance = this->varSamplingDistance()->getData();

		if (sampling_distance < EPSILON)
			sampling_distance = 0.005f;

		auto center = this->varLocation()->getData();
		auto scale = this->varScale()->getData();

		auto quat = this->computeQuaternion();

		Transform<float, 3> tr(center, quat.toMatrix3x3(), scale);

		std::vector<Vec3f> pos_list;
		std::vector<Vec3f> vel_list;

		Vec3f v0 = this->varVelocityMagnitude()->getData()*quat.rotate(Vec3f(0, -1, 0));

		auto w = 0.5 * this->varWidth()->getData();
		auto h = 0.5 * this->varHeight()->getData();

		for (float x = -w; x <= w; x += sampling_distance)
		{
			for (float z = -h; z <= h; z += sampling_distance)
			{
				Vec3f p = Vec3f(x, 0, z);
				if (rand() % 5 == 0)
				{
					pos_list.push_back(tr * p);
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
	
	void SquareEmitter::tranformChanged()
	{
		std::vector<Vec3f> vertices;
		std::vector<TopologyModule::Edge> edges;

		auto center = this->varLocation()->getData();
		auto scale = this->varScale()->getData();

		auto quat = this->computeQuaternion();

		auto w = this->varWidth()->getData();
		auto h = this->varHeight()->getData();

		Transform<float, 3> tr(Coord(0), quat.toMatrix3x3(), scale);

		auto Nx = tr * Vec3f(0.5 * w, 0, 0);
		auto Nz = tr * Vec3f(0, 0, 0.5 * h);

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

	void SquareEmitter::resetStates()
	{
		tranformChanged();
	}
}