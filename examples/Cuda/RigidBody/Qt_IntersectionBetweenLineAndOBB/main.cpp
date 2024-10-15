/**
 * Copyright 2023 Xiaowei He
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <QtApp.h>

#include <SceneGraph.h>

#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

#include "Module/ComputeModule.h"

#include "BasicShapes/CubeModel.h"

#include <Mapping/DiscreteElementsToTriangleSet.h>

using namespace std;
using namespace dyno;

/**
 * This example demonstrates how to compute and visualize the intersection between a segment and an oriented bounding box
 */


template<typename TDataType>
class Segement : public ParametricModel<TDataType>
{
	DECLARE_TCLASS(Segement, TDataType);
public:
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;

	Segement()
	{
		this->stateEdgeSet()->allocate();

		auto pointRender = std::make_shared<GLPointVisualModule>();
		pointRender->varPointSize()->setValue(0.02);
		pointRender->varBaseColor()->setValue(Color(0.0f, 1.0f, 0.0f));
		this->stateEdgeSet()->connect(pointRender->inPointSet());
		this->graphicsPipeline()->pushModule(pointRender);

		auto lineRender = std::make_shared<GLWireframeVisualModule>();
		lineRender->varRenderMode()->getDataPtr()->setCurrentKey(GLWireframeVisualModule::CYLINDER);
		this->stateEdgeSet()->connect(lineRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(lineRender);

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&Segement<TDataType>::valueChanged, this));

		this->varLocation()->attach(callback);
		this->varRotation()->attach(callback);
		this->varScale()->attach(callback);
	}

public:
	DEF_VAR(Real, Length, Real(2), "Line length");

	DEF_INSTANCE_STATE(EdgeSet<TDataType>, EdgeSet, "");

	DEF_VAR_OUT(TSegment3D<Real>, Segment, "");

protected:
	void resetStates() override {
		this->varLocation()->setValue(Coord(0));
		this->varRotation()->setValue(Coord(0));
		this->varScale()->setValue(Coord(1));

		valueChanged();
	}

	void valueChanged() {
		auto center = this->varLocation()->getData();
		auto rot = this->varRotation()->getData();
		auto scale = this->varScale()->getData();

		Real len = this->varLength()->getData();

		Quat<Real> q = Quat<Real>(M_PI * rot[0] / 180, Coord(1, 0, 0))
			* Quat<Real>(M_PI * rot[1] / 180, Coord(0, 1, 0))
			* Quat<Real>(M_PI * rot[2] / 180, Coord(0, 0, 1));

		//A lambda function to rotate a vertex
		auto RV = [&](const Coord& v)->Coord {
			return center + q.rotate(v - center);
		};

		Coord from = RV(center - Coord(0.5 * len, 0, 0));
		Coord to = RV(center + Coord(0.5 * len, 0, 0));

		auto edgeSet = this->stateEdgeSet()->getDataPtr();

		std::vector<Coord> hPos;
		std::vector<TopologyModule::Edge> edges;
		hPos.push_back(from);
		hPos.push_back(to);
		edges.push_back(TopologyModule::Edge(0, 1));

		edgeSet->setPoints(hPos);
		edgeSet->setEdges(edges);

		this->outSegment()->setValue(TSegment3D<Real>(from, to));

		hPos.clear();
	};

private:
};

IMPLEMENT_TCLASS(Segement, TDataType);


template<typename TDataType>
class ComputeIntersectionModule : public ComputeModule
{
	DECLARE_TCLASS(ComputeDistanceModule, TDataType);
public:
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;

	ComputeIntersectionModule() {};
	
public:
	DEF_VAR_IN(TSegment3D<Real>, Segment, "");

	DEF_VAR_IN(TOrientedBox3D<Real>, Cube, "");

	DEF_INSTANCE_OUT(EdgeSet<TDataType>, EdgeSet, "");
	
protected:
	void compute() override {
		auto segment = this->inSegment()->getData();
		auto cube = this->inCube()->getData();

		TSegment3D<Real> inter;
		int num = segment.intersect(cube,inter);

		if (this->outEdgeSet()->isEmpty()){
			this->outEdgeSet()->allocate();
		}

		std::vector<Coord> coords;
		std::vector<TopologyModule::Edge> edges;

		if (num == 2)
		{
			coords.push_back(inter.v0);
			coords.push_back(inter.v1);
			edges.push_back(TopologyModule::Edge(0, 1));

		}else if (num == 1)
		{
			coords.push_back(inter.v0);
		}

		auto edgeSet = this->outEdgeSet()->getDataPtr();
		edgeSet->setPoints(coords);
		edgeSet->setEdges(edges);

		coords.clear();
		edges.clear();
	};

private:

};

IMPLEMENT_TCLASS(ComputeIntersectionModule, TDataType);

template<typename TDataType>
class Intersection : public Node
{
	DECLARE_TCLASS(Intersection, TDataType);
public:
	typedef typename TDataType::Coord Coord;

	Intersection() {
		auto computeDistance = std::make_shared<ComputeIntersectionModule<TDataType>>();
		this->inSegment()->connect(computeDistance->inSegment());
		this->inCube()->connect(computeDistance->inCube());
		this->graphicsPipeline()->pushModule(computeDistance);

		auto pointRender = std::make_shared<GLPointVisualModule>();
		pointRender->varPointSize()->setValue(0.025);
		pointRender->varBaseColor()->setValue(Color(1.0f, 0.0f, 0.0f));
		computeDistance->outEdgeSet()->connect(pointRender->inPointSet());
		this->graphicsPipeline()->pushModule(pointRender);

		auto wireRender = std::make_shared<GLWireframeVisualModule>();
		wireRender->varRenderMode()->getDataPtr()->setCurrentKey(GLWireframeVisualModule::CYLINDER);
		computeDistance->outEdgeSet()->connect(wireRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(wireRender);
	};

	DEF_VAR_IN(TSegment3D<Real>, Segment, "");

	DEF_VAR_IN(TOrientedBox3D<Real>, Cube, "");

private:

};

IMPLEMENT_TCLASS(Intersection, TDataType);

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto point = scn->addNode(std::make_shared<Segement<DataType3f>>());

	auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());

	auto calculation = scn->addNode(std::make_shared<Intersection<DataType3f>>());

	point->outSegment()->connect(calculation->inSegment());
	cube->outCube()->connect(calculation->inCube());

	return scn;
}

int main()
{
	QtApp app;
	app.setSceneGraph(createScene());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


