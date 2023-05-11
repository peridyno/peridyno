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

#include "CubeModel.h"

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Collision/CollisionDetectionAlgorithm.h>

using namespace std;
using namespace dyno;

/**
 * This example demonstrates how to compute the contact manifold between two OBBs
 */


template<typename TDataType>
class ComputeContactManifold : public ComputeModule
{
	DECLARE_TCLASS(ComputeContactManifold, TDataType);
public:
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;

	ComputeContactManifold() {};
	
	DEF_VAR_IN(TOrientedBox3D<Real>, CubeA, "");

	DEF_VAR_IN(TOrientedBox3D<Real>, CubeB, "");

	DEF_INSTANCE_OUT(EdgeSet<TDataType>, EdgeSet, "");
	
protected:
	void compute() override {
		auto cubeA = this->inCubeA()->getData();
		auto cubeB = this->inCubeB()->getData();

		TManifold<Real> manifold;

		CollisionDetection<Real>::request(manifold, cubeA, cubeB);

		if (this->outEdgeSet()->isEmpty()){
			this->outEdgeSet()->allocate();
		}

		std::vector<Coord> vertices;
		std::vector<TopologyModule::Edge> edges;


		uint num = manifold.contactCount;
		for (uint i = 0; i < num; i++)
		{
			vertices.push_back(manifold.contacts[i].position);
			edges.push_back(TopologyModule::Edge(i, (i + 1) % num));
		}

		auto edgeSet = this->outEdgeSet()->getDataPtr();
		edgeSet->setPoints(vertices);
		edgeSet->setEdges(edges);

		vertices.clear();
		edges.clear();
	};

private:

};

IMPLEMENT_TCLASS(ComputeContactManifold, TDataType);

template<typename TDataType>
class SAT : public Node
{
	DECLARE_TCLASS(SAT, TDataType);
public:
	typedef typename TDataType::Coord Coord;

	SAT() {

		auto computeContacts= std::make_shared<ComputeContactManifold<TDataType>>();
		this->inCubeA()->connect(computeContacts->inCubeA());
		this->inCubeB()->connect(computeContacts->inCubeB());
		this->graphicsPipeline()->pushModule(computeContacts);

		auto pointRender = std::make_shared<GLPointVisualModule>();
		pointRender->varPointSize()->setValue(0.02);
		pointRender->varBaseColor()->setValue(Color(1.0f, 0.0f, 0.0f));
		computeContacts->outEdgeSet()->connect(pointRender->inPointSet());
		this->graphicsPipeline()->pushModule(pointRender);

		auto wireRender = std::make_shared<GLWireframeVisualModule>();
		wireRender->varRenderMode()->getDataPtr()->setCurrentKey(GLWireframeVisualModule::CYLINDER);
		computeContacts->outEdgeSet()->connect(wireRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(wireRender);
	};

	DEF_VAR_IN(TOrientedBox3D<Real>, CubeA, "");

	DEF_VAR_IN(TOrientedBox3D<Real>, CubeB, "");

private:

};

IMPLEMENT_TCLASS(SAT, TDataType);

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto cubeA = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cubeA->varLocation()->setValue(Vec3f(0.6f, 0.5f, 0.0f));

	auto cubeB = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cubeB->varLocation()->setValue(Vec3f(-0.6f, 0.5f, 0.0f));

	auto sat = scn->addNode(std::make_shared<SAT<DataType3f>>());

	cubeA->outCube()->connect(sat->inCubeA());
	cubeB->outCube()->connect(sat->inCubeB());

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


