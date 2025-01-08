#include "PoissonEmitter.h"
#include "GLWireframeVisualModule.h"

#include <time.h>

#include <stdlib.h>

namespace dyno
{
	template<typename TDataType>
	PoissonEmitter<TDataType>::PoissonEmitter()
		: ParticleEmitter<TDataType>()
	{
		srand(time(0));

		this->varWidth()->setRange(0.01, 10.0f);
		this->varHeight()->setRange(0.01, 10.0f);

		this->varSamplingDistance()->setValue(0.008f);

		this->stateOutline()->setDataPtr(std::make_shared<EdgeSet<TDataType>>());

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&PoissonEmitter<TDataType>::tranformChanged, this));

		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);

		this->varWidth()->attach(callback);
		this->varHeight()->attach(callback);

		mPlane = std::make_shared< PoissonPlane<TDataType>>();

		auto wireRender = std::make_shared<GLWireframeVisualModule>();
		wireRender->setColor(Color(0, 1, 0));
		this->stateOutline()->connect(wireRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(wireRender);
	}

	template<typename TDataType>
	PoissonEmitter<TDataType>::~PoissonEmitter()
	{
	}

	

	template<typename TDataType>
	void PoissonEmitter<TDataType>::generateParticles()
	{
		mCounter++;
		uint delayStart = this->varDelayStart()->getValue();	
		//std::cout << delayStart << ", " << mCounter << std::endl;
		if (delayStart > mCounter)
		{
			this->mPosition.reset();
			this->mVelocity.reset();
			return;
		}


		//std::cout << mPlane->getPoints().size() << std::endl;

		std::vector<Vec2f> temp_points = mPlane->getPoints();

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

		//mPlane->varUpper()->setValue(Vec2f(w, h));
		//mPlane->varLower()->setValue(Vec2f(-w, -h));
		//mPlane->compute();
		//mPlane->varSamplingDistance()->setValue(this->varSamplingDistance()->getValue());

		if (this->varEmitterShape()->getData() == EmitterShape::Round)
		{
			Real radius = w < h ? w : h;

			mPlane->varUpper()->setValue(Vec2f(radius, radius));
			mPlane->varLower()->setValue(Vec2f(-radius, -radius));
			mPlane->compute();
			mPlane->varSamplingDistance()->setValue(this->varSamplingDistance()->getValue());


			for (int i = 0; i < temp_points.size(); i++)
			{
				if (sqrt(temp_points[i][0] * temp_points[i][0] + temp_points[i][1] * temp_points[i][1]) < radius)
				{
					Coord pos_temp = Coord(temp_points[i][0], 0.0, temp_points[i][1]);
					pos_list.push_back(tr * pos_temp);
					vel_list.push_back(v0);
				}
			}
		}

		else
		{
			mPlane->varUpper()->setValue(Vec2f(w, h));
			mPlane->varLower()->setValue(Vec2f(-w, -h));
			mPlane->compute();
			mPlane->varSamplingDistance()->setValue(this->varSamplingDistance()->getValue());


			for (int i = 0; i < temp_points.size(); i++)
			{
				Coord pos_temp = Coord(temp_points[i][0], 0.0, temp_points[i][1]);
				pos_list.push_back(tr * pos_temp);
				vel_list.push_back(v0);
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
	void PoissonEmitter<TDataType>::tranformChanged()
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
	void PoissonEmitter<TDataType>::resetStates()
	{
		tranformChanged();
		mCounter = 0;

	}

	DEFINE_CLASS(PoissonEmitter);
}