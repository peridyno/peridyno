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

#include <initializeModeling.h>

#include "Module/ComputeModule.h"

#include "CubeModel.h"
#include "CollisionDetector.h"

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Collision/CollisionDetectionAlgorithm.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto cubeA = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cubeA->varLocation()->setValue(Vec3f(0.6f, 0.5f, 0.0f));

	auto cubeB = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cubeB->varLocation()->setValue(Vec3f(-0.6f, 0.5f, 0.0f));

	auto sat = scn->addNode(std::make_shared<CollisionDetector<DataType3f>>());

	cubeA->connect(sat->importShapeA());
	cubeB->connect(sat->importShapeB());

	return scn;
}

int main()
{
	Modeling::initStaticPlugin();

	QtApp app;
	app.setSceneGraph(createScene());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}


