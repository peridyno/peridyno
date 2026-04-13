#include "CircleModel2D.h"

#include "GLWireframeVisualModule.h"

namespace dyno
{
	template<typename TDataType>
	CircleModel2D<TDataType>::CircleModel2D()
		: BasicShape2D<TDataType>()
	{
		this->stateEdgeSet()->allocate();

		this->varRadius()->setRange(0.001f, 100.0f);

		this->varLocation()->attach(std::make_shared<FCallBackFunc>(
			[=]() {
				auto loc = this->varLocation()->getValue();
				this->varCenter()->setValue(Coord2D(loc.x, loc.y));
			}
		));

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&CircleModel2D<TDataType>::varChanged, this));

		this->varRadius()->attach(callback);
		this->varCenter()->attach(callback);
		this->varSegmentNumber()->attach(callback);

		auto esRender = std::make_shared<GLWireframeVisualModule>();
		this->stateEdgeSet()->connect(esRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(esRender);

		this->stateEdgeSet()->promoteOuput();
		this->stateCircle()->promoteOuput();
	}


	template<typename TDataType>
	void CircleModel2D<TDataType>::resetStates()
	{
		varChanged();
	}

	template<typename TDataType>
	void CircleModel2D<TDataType>::varChanged()
	{
		auto r = this->varRadius()->getValue();
		auto center = this->varCenter()->getValue();

		auto num = this->varSegmentNumber()->getValue();

		auto q = this->computeQuaternion();

		//A lambda function to rotate a vertex
		auto RV = [&](const Coord3D& v)->Coord3D {
			return Coord3D(center.x, center.y, 0) + q.rotate(v);
			};

		std::vector<Coord3D> vertices;
		std::vector<Topology::Edge> edges;

		Real deltaTheta = 2 * M_PI / num;

		Real theta = 0;
		for (uint i = 0; i < num; i++)
		{
			theta += deltaTheta;

			Real x = r * std::sin(theta);
			Real y = r * std::cos(theta);

			vertices.push_back(RV(Coord3D(x, y, 0)));
			edges.push_back(Topology::Edge(i, (i + 1) % num));
		}

		auto es = this->stateEdgeSet()->getDataPtr();
		es->setPoints(vertices);
		es->setEdges(edges);
		es->update();

		Circle2D circle;
		circle.center = center;
		circle.radius = r;

		this->stateCircle()->setValue(circle);

		vertices.clear();
		edges.clear();
	}

	DEFINE_CLASS(CircleModel2D);
}
