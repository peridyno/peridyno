#include "RectangleModel2D.h"

#include "GLWireframeVisualModule.h"

namespace dyno
{
	template<typename TDataType>
	RectangleModel2D<TDataType>::RectangleModel2D()
		: BasicShape2D<TDataType>()
	{
		this->stateEdgeSet()->allocate();

		this->varWidth()->setRange(0.001f, 100.0f);
		this->varHeight()->setRange(0.001f, 100.0f);

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&RectangleModel2D<TDataType>::varChanged, this));

		this->varLocation()->attach(callback);
		this->varRotation()->attach(callback);
		this->varWidth()->attach(callback);
		this->varHeight()->attach(callback);

		auto esRender = std::make_shared<GLWireframeVisualModule>();
		this->stateEdgeSet()->connect(esRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(esRender);

		this->stateEdgeSet()->promoteOuput();
		this->stateRectangle()->promoteOuput();
	}

	template<typename TDataType>
	void RectangleModel2D<TDataType>::resetStates()
	{
		varChanged();
	}

	template<typename TDataType>
	void RectangleModel2D<TDataType>::varChanged()
	{
		auto center = this->varLocation()->getValue();
		auto rot = this->varRotation()->getValue();
		auto scale = this->varScale()->getValue();

		auto w = this->varWidth()->getValue();
		auto h = this->varHeight()->getValue();

		this->varLocation()->setValue(Coord3D(center[0], center[1], 0), false);
		this->varRotation()->setValue(Coord3D(0, 0, rot[1]), false);

		auto q = this->computeQuaternion();

		//A lambda function to rotate a vertex
		auto RV = [&](const Coord& v)->Coord {
			return center + q.rotate(v);
			};

		std::vector<Coord> vertices;
		vertices.push_back(RV(Coord3D(w / 2, h / 2, 0)));
		vertices.push_back(RV(Coord3D(-w / 2, h / 2, 0)));
		vertices.push_back(RV(Coord3D(-w / 2, -h / 2, 0)));
		vertices.push_back(RV(Coord3D(w / 2, -h / 2, 0)));

		std::vector<Topology::Edge> edges;
		edges.push_back(Topology::Edge(0, 1));
		edges.push_back(Topology::Edge(1, 2));
		edges.push_back(Topology::Edge(2, 3));
		edges.push_back(Topology::Edge(3, 0));

		auto edgeSet = this->stateEdgeSet()->getDataPtr();
		edgeSet->setPoints(vertices);
		edgeSet->setEdges(edges);
		edgeSet->update();

		auto u_rot = RV(Coord3D(w / 2, 0, 0));
		auto v_rot = RV(Coord3D(0, h / 2, 0));

		TOrientedBox2D<Real> obb2d;
		obb2d.center = Coord2D(center[0], center[1]);
		obb2d.extent = Coord2D(w / 2, h / 2);
		obb2d.u = Coord2D(u_rot[0], u_rot[1]);
		obb2d.v = Coord2D(v_rot[0], v_rot[1]);

		this->stateRectangle()->setValue(obb2d);

		vertices.clear();
		edges.clear();
	}

	DEFINE_CLASS(RectangleModel2D);
}