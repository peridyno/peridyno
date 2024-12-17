#include "CircularEmitter.h"
#include <time.h>

#include <stdlib.h>

namespace dyno
{
	template<typename TDataType>
	CircularEmitter<TDataType>::CircularEmitter()
		: ParticleEmitter<TDataType>()
	{
		srand(time(0));

		this->varRadius()->setRange(0.0, 10.0);

		this->stateOutline()->setDataPtr(std::make_shared<EdgeSet<TDataType>>());

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&CircularEmitter<TDataType>::tranformChanged, this));

		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);

		this->varRadius()->attach(callback);
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

		auto center = this->varLocation()->getData();
		auto scale = this->varScale()->getData();
		auto quat = this->computeQuaternion();

		auto r = this->varRadius()->getData();

		std::vector<Coord> pos_list;
		std::vector<Coord> vel_list;

		Real velMag = this->varVelocityMagnitude()->getData();
		Coord v0 = velMag * quat.rotate(Vec3f(0, -1, 0));

		Transform<Real, 3> tr(center, quat.toMatrix3x3(), scale);

		Real a = r * scale.x;
		Real b = r * scale.z;

		Real invA2 = Real(1) / (a * a);
		Real invB2 = Real(1) / (b * b);

		for (Real x = -a; x <= a; x += sampling_distance)
		{
			for (Real z = -b; z <= b; z += sampling_distance)
			{
				if ((x * x * invA2 + z * z * invB2) < 1 && rand() % 5 == 0)
				{
					Coord p = Coord(x / scale.x, 0, z / scale.z);

					pos_list.push_back(tr * p);
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
	void CircularEmitter<TDataType>::tranformChanged()
	{
		std::vector<Coord> vertices;
		std::vector<TopologyModule::Edge> edges;

		auto center = this->varLocation()->getData();
		auto scale = this->varScale()->getData();
		auto quat = this->computeQuaternion();

		auto r = this->varRadius()->getData();

		Transform<Real, 3> tr(center, quat.toMatrix3x3(), scale);

		int segNum = 36;
		Real deltaTheta = 2 * M_PI / segNum;

		for (int i = 0; i < segNum; i++)
		{
			Real x = r * sin(i * deltaTheta);
			Real z = r * cos(i * deltaTheta);

			vertices.push_back(tr * Coord(x, 0, z));
			edges.push_back(TopologyModule::Edge(i, (i + 1) % segNum));
		}

		auto edgeTopo = this->stateOutline()->getDataPtr();

		edgeTopo->setPoints(vertices);
		edgeTopo->setEdges(edges);

		vertices.clear();
		edges.clear();
	}

	template<typename TDataType>
	void CircularEmitter<TDataType>::resetStates()
	{
		tranformChanged();
	}

	DEFINE_CLASS(CircularEmitter);
}