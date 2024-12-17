#include "SquareEmitter.h"
#include <time.h>

#include <stdlib.h>

namespace dyno
{
	template<typename TDataType>
	SquareEmitter<TDataType>::SquareEmitter()
		: ParticleEmitter<TDataType>()
	{
		srand(time(0));

		this->varWidth()->setRange(0.01, 10.0f);
		this->varHeight()->setRange(0.01, 10.0f);

		this->stateOutline()->setDataPtr(std::make_shared<EdgeSet<TDataType>>());

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&SquareEmitter<TDataType>::tranformChanged, this));

		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);

		this->varWidth()->attach(callback);
		this->varHeight()->attach(callback);
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
		auto scale = this->varScale()->getData();

		auto quat = this->computeQuaternion();

		Transform<Real, 3> tr(center, quat.toMatrix3x3(), scale);

		std::vector<Coord> pos_list;
		std::vector<Coord> vel_list;

		Coord v0 = this->varVelocityMagnitude()->getData()*quat.rotate(Vec3f(0, -1, 0));

		auto w = 0.5 * this->varWidth()->getData();
		auto h = 0.5 * this->varHeight()->getData();

		for (Real x = -w; x <= w; x += sampling_distance)
		{
			for (Real z = -h; z <= h; z += sampling_distance)
			{
				Coord p = Coord(x, 0, z);
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
	
	template<typename TDataType>
	void SquareEmitter<TDataType>::tranformChanged()
	{
		std::vector<Coord> vertices;
		std::vector<TopologyModule::Edge> edges;

		auto center = this->varLocation()->getData();
		auto scale = this->varScale()->getData();

		auto quat = this->computeQuaternion();

		auto w = this->varWidth()->getData();
		auto h = this->varHeight()->getData();

		Transform<Real, 3> tr(Coord(0), quat.toMatrix3x3(), scale);

		auto Nx = tr * Coord(0.5 * w, 0, 0);
		auto Nz = tr * Coord(0, 0, 0.5 * h);

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


	template<typename TDataType>
	void SquareEmitter<TDataType>::resetStates()
	{
		tranformChanged();
	}

	DEFINE_CLASS(SquareEmitter);
}