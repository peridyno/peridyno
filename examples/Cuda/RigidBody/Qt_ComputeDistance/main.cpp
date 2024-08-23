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
 * This example demonstrates how to compute and visualize the distance between a point and an oriented bounding box
 */


template<typename TDataType>
class Point : public ParametricModel<TDataType>
{
	DECLARE_TCLASS(Point, TDataType);
public:
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;

	Point()
	{
		this->statePointSet()->allocate();

		auto pointRender = std::make_shared<GLPointVisualModule>();
		pointRender->varPointSize()->setValue(0.025);
		this->statePointSet()->connect(pointRender->inPointSet());
		this->graphicsPipeline()->pushModule(pointRender);

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&Point<TDataType>::locationChanged, this));
		this->varLocation()->attach(callback);

		this->varLocation()->setValue(Coord(2.0, 0.0, 0.0));
	}

public:
	DEF_INSTANCE_STATE(PointSet<TDataType>, PointSet, "");

	DEF_VAR_OUT(TPoint3D<Real>, Point, "");

private:
	void locationChanged() {
		auto loc = this->varLocation()->getValue();
		
		auto points = this->statePointSet()->getDataPtr();

		std::vector<Coord> hPos;
		hPos.push_back(loc);

		points->setPoints(hPos);

		this->outPoint()->setValue(TPoint3D<Real>(loc));

		hPos.clear();
	};

private:
};

IMPLEMENT_TCLASS(Point, TDataType);


template<typename TDataType>
class ComputeDistanceModule : public ComputeModule
{
	DECLARE_TCLASS(ComputeDistanceModule, TDataType);
public:
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;

	ComputeDistanceModule() {};
	
	DEF_VAR_IN(TPoint3D<Real>, Point, "");

	DEF_VAR_IN(TOrientedBox3D<Real>, Cube, "");

	DEF_INSTANCE_OUT(EdgeSet<TDataType>, EdgeSet, "");
	
protected:
	void compute() override {
		auto point = this->inPoint()->getData();
		auto cube = this->inCube()->getData();

		Bool inside;
		TPoint3D<Real> q = point.project(cube, inside);

		if (this->outEdgeSet()->isEmpty()){
			this->outEdgeSet()->allocate();
		}

		std::vector<Coord> coords;
		std::vector<TopologyModule::Edge> edges;
		coords.push_back(point.origin);
		coords.push_back(q.origin);
		edges.push_back(TopologyModule::Edge(0, 1));

		auto edgeSet = this->outEdgeSet()->getDataPtr();
		edgeSet->setPoints(coords);
		edgeSet->setEdges(edges);

		coords.clear();
		edges.clear();
	};

private:

};

IMPLEMENT_TCLASS(ComputeDistanceModule, TDataType);

template<typename TDataType>
class Distance : public Node
{
	DECLARE_TCLASS(Distance, TDataType);
public:
	typedef typename TDataType::Coord Coord;

	Distance() {

		auto computeDistance = std::make_shared<ComputeDistanceModule<TDataType>>();
		this->inPoint()->connect(computeDistance->inPoint());
		this->inCube()->connect(computeDistance->inCube());
		this->graphicsPipeline()->pushModule(computeDistance);

		auto pointRender = std::make_shared<GLPointVisualModule>();
		pointRender->varPointSize()->setValue(0.02);
		pointRender->varBaseColor()->setValue(Color(1.0f, 0.0f, 0.0f));
		computeDistance->outEdgeSet()->connect(pointRender->inPointSet());
		this->graphicsPipeline()->pushModule(pointRender);

		auto wireRender = std::make_shared<GLWireframeVisualModule>();
		wireRender->varRenderMode()->getDataPtr()->setCurrentKey(GLWireframeVisualModule::CYLINDER);
		computeDistance->outEdgeSet()->connect(wireRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(wireRender);
	};

	DEF_VAR_IN(TPoint3D<Real>, Point, "");

	DEF_VAR_IN(TOrientedBox3D<Real>, Cube, "");

private:

};

IMPLEMENT_TCLASS(Distance, TDataType);

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto point = scn->addNode(std::make_shared<Point<DataType3f>>());

	auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());

	auto calculation = scn->addNode(std::make_shared<Distance<DataType3f>>());

	point->outPoint()->connect(calculation->inPoint());
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


