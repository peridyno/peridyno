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

		this->varLocation2D()->attach(callback);
		this->varRotation2D()->attach(callback);
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
		auto scale = this->varScale2D()->getValue();

		auto w = this->varWidth()->getValue();
		auto h = this->varHeight()->getValue();

		TOrientedBox2D<Real> obb2d;
		obb2d.center = this->varLocation2D()->getValue();
		obb2d.extent = Coord2D(scale[0] * w / 2, scale[1] * h / 2);
		obb2d.u = this->computeRotate(Coord2D(1, 0));
		obb2d.v = this->computeRotate(Coord2D(0, 1));
		this->stateRectangle()->setValue(obb2d);

		Coord3D center, width_half, height_half;
		if (this->varVisPlane()->getValue() == 0)
		{
			center = Coord3D(obb2d.center[0], obb2d.center[1], 0);
			width_half = Coord3D(obb2d.extent[0] * obb2d.u[0], obb2d.extent[0] * obb2d.u[1], 0);
			height_half = Coord3D(obb2d.extent[1] * obb2d.v[0], obb2d.extent[1] * obb2d.v[1], 0);
		}
		else if (this->varVisPlane()->getValue() == 1)
		{
			center = Coord3D(obb2d.center[0], 0, obb2d.center[1]);
			width_half = Coord3D(obb2d.extent[0] * obb2d.u[0], 0, obb2d.extent[0] * obb2d.u[1]);
			height_half = Coord3D(obb2d.extent[1] * obb2d.v[0], 0, obb2d.extent[1] * obb2d.v[1]);
		}

		std::vector<Coord3D> vertices;
		vertices.push_back(center + width_half + height_half);
		vertices.push_back(center - width_half + height_half);
		vertices.push_back(center - width_half - height_half);
		vertices.push_back(center + width_half - height_half);

		std::vector<Topology::Edge> edges;
		edges.push_back(Topology::Edge(0, 1));
		edges.push_back(Topology::Edge(1, 2));
		edges.push_back(Topology::Edge(2, 3));
		edges.push_back(Topology::Edge(3, 0));

		auto edgeSet = this->stateEdgeSet()->getDataPtr();
		edgeSet->setPoints(vertices);
		edgeSet->setEdges(edges);
		edgeSet->update();

		vertices.clear();
		edges.clear();
	}

	DEFINE_CLASS(RectangleModel2D);
}