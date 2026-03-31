#include "FadeOutEmitter.h"
#include "GLWireframeVisualModule.h"

#include <time.h>

#include <stdlib.h>

namespace dyno
{
	template<typename TDataType>
	FadeOutEmitter<TDataType>::FadeOutEmitter()
		: ParticleEmitter<TDataType>()
	{
		srand(time(0));

		this->varWidth()->setRange(0.01, 10.0f);
		this->varHeight()->setRange(0.01, 10.0f);

		this->stateOutline()->setDataPtr(std::make_shared<EdgeSet<TDataType>>());

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&FadeOutEmitter<TDataType>::tranformChanged, this));

		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);

		this->varWidth()->attach(callback);
		this->varHeight()->attach(callback);

		auto wireRender = std::make_shared<GLWireframeVisualModule>();
		wireRender->varBaseColor()->setValue(Color(0, 1, 0));
		this->stateOutline()->connect(wireRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(wireRender);
	}

	template<typename TDataType>
	FadeOutEmitter<TDataType>::~FadeOutEmitter()
	{
	}

	template<typename TDataType>
	void FadeOutEmitter<TDataType>::generateParticles()
	{

		frame_++;

		auto sampling_distance = this->varSamplingDistance()->getData();

		if (sampling_distance < EPSILON)
			sampling_distance = Real(0.005);

		auto center = this->varLocation()->getData();
		auto scale = this->varScale()->getData();

		auto quat = this->computeQuaternion();

		uint f_num = this->stateFrameNumber()->getValue();

		my_time_ = this->getDt() * (Real)(f_num);


		if (my_time_ * this->varMovingVelocity()->getData()[2] < 0.2)
		{
			my_position = my_time_ * this->varMovingVelocity()->getData();
		}

		std::cout <<"My pos:"  << my_position << std::endl;

		//std::cout <<  "!!!!!!! - Frame:" << this->stateFrameNumber()->getValue() << "My Time:" << my_time_ << "My Position:" << my_position << std::endl;

		Transform<Real, 3> tr(center, quat.toMatrix3x3(), scale);

		std::vector<Coord> pos_list;
		std::vector<Coord> vel_list;

		Coord v0 = this->varVelocityMagnitude()->getData() * quat.rotate(Vec3f(0, -1, 0));

		auto w = 0.5 * this->varWidth()->getData();
		auto h = 0.5 * this->varHeight()->getData();

		
		Real radius = std::max(w, h)*2.0;
	
		Real a = (Real)(this->varFadeOutEnd()->getData() - this->varFadeOutBegin()->getData());
		Real b = (Real)((Real)(this->varFadeOutEnd()->getData()) - (Real)(frame_));
		b = b < 0 ? 0 : b;
		Real t = (b / a) * radius;

		Coord movingCenter_ = my_position + center;

		std::cout << "Emitter MMM: " << t << " My Position " << my_position << std::endl;

		if (frame_ < this->varFadeOutBegin()->getData()) {

			for (Real x = -w; x <= w; x += sampling_distance)
			{
				for (Real z = -h; z <= h; z += sampling_distance)
				{
					Coord p = Coord(x, 0, z);
					{
						pos_list.push_back(tr * p + my_position);
						vel_list.push_back(v0);
					}
				}
			}
		}
		else //if(frame_ < this->varFadeOutEnd()->getData())
		{
			for (Real x = -w; x <= w; x += sampling_distance)
			{
				for (Real z = -h; z <= h; z += sampling_distance)
				{
					Coord p = Coord(x, 0, z);
					Coord pos_ = tr * p + my_position;
					if (t < 0.010) t = 0.010;
					//if ((pos_ - center).norm() < t)
					if ((pos_ - movingCenter_).norm() < t)
					{
						
						{
							pos_list.push_back(pos_);
							vel_list.push_back(v0);
						}
					}


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
	void FadeOutEmitter<TDataType>::tranformChanged()
	{
		std::vector<Coord> vertices;
		std::vector<Topology::Edge> edges;

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

		edges.push_back(Topology::Edge(0, 1));
		edges.push_back(Topology::Edge(1, 2));
		edges.push_back(Topology::Edge(2, 3));
		edges.push_back(Topology::Edge(3, 0));

		auto edgeTopo = this->stateOutline()->getDataPtr();

		edgeTopo->setPoints(vertices);
		edgeTopo->setEdges(edges);

		vertices.clear();
		edges.clear();
	}


	template<typename TDataType>
	void FadeOutEmitter<TDataType>::resetStates()
	{
		ParticleEmitter<TDataType>::resetStates();

		frame_ = 0;

		my_time_ = 0.0f;

		my_position = Coord(0.0);

		tranformChanged();
	}

	DEFINE_CLASS(FadeOutEmitter);
}