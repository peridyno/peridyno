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

#include "DataTypes.h"
#include "Module/ComputeModule.h"

#include "BasicShapes/CubeModel.h"
#include "BasicShapes/SphereModel.h"
#include "BasicShapes/CapsuleModel.h"
#include "BasicShapes/MedialConeModel.h"
#include "BasicShapes/MedialSlabModel.h"
#include "CollisionDetector.h"

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Collision/CollisionDetectionAlgorithm.h>
#include <memory>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto cube = [&](Vec3f x)
	{
		auto c = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
		c->varLocation()->setValue(x);
		return c;
	};

	auto sphere = [&](Vec3f x, float r)
	{
		auto s = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
		s->varLocation()->setValue(x);
		s->varRadius()->setValue(r);
		return s;
	};

	auto capsule = [&](Vec3f x, Vec3f rot, float r, float h)
	{
		auto c = scn->addNode(std::make_shared<CapsuleModel<DataType3f>>());
		c->varLocation()->setValue(x);
		c->varRadius()->setValue(r);
		c->varHeight()->setValue(h);
		c->varRotation()->setValue(rot);
		return c;
	};

	auto medialSlab = [&](Vec3f center, Vec3f pA, Vec3f pB, Vec3f pC, float r1, float r2, float r3)
		{
			auto c = scn->addNode(std::make_shared<MedialSlabModel<DataType3f>>());
			c->varLocation()->setValue(center);
			c->varPointA()->setValue(pA);
			c->varPointB()->setValue(pB);
			c->varPointC()->setValue(pC);
			c->varRadiusA()->setValue(r1);
			c->varRadiusB()->setValue(r2);
			c->varRadiusC()->setValue(r3);
			return c;
		};

	auto medialCone = [&](Vec3f center, Vec3f dir, float halfLength, float r1, float r2)
		{
			auto c = scn->addNode(std::make_shared<MedialConeModel<DataType3f>>());
			c->varLocation()->setValue(center);
			c->varPointA()->setValue(dir * halfLength);
			c->varPointB()->setValue(-dir * halfLength);
			c->varRadiusA()->setValue(r1);
			c->varRadiusB()->setValue(r2);
			return c;
		};

	std::shared_ptr<BasicShape<DataType3f>> cubes[4], spheres[4], capsules[4];
	std::shared_ptr<BasicShape<DataType3f>> medalCones[2];
	std::shared_ptr<BasicShape<DataType3f>> medalslabs[2];
	
	medalCones[0] = medialCone(Vec3f(0, 0, 0), Vec3f(1, 0, 0), 0.5, 0.2, 0.4);
	medalCones[1] = medialCone(Vec3f(0, 0.2, 0), Vec3f(-1, 0, 0), 0.5, 0.2, 0.4);
	medalslabs[0] = medialSlab(Vec3f(0, 0, 0), Vec3f(1, 0, 0), Vec3f(-1, 0, 0), Vec3f(0, 0, 1), 0.2, 0.5, 1.0);
	medalslabs[1] = medialSlab(Vec3f(0, 0, 0), Vec3f(1, 0, 0), Vec3f(-1, 0, 0), Vec3f(0, 0, 1), 0.5, 0.5, 1.0);
	cubes[0] = cube(Vec3f(1.478f, 0.5f, 0.0f));
	cubes[1] = cube(Vec3f(0.598f, 0.932f, -0.737f));
	// cubes[2] = cube(Vec3f(-0.6f, 0.5f, 0.0f));
	// cubes[3] = cube(Vec3f(-0.6f, 0.5f, 0.0f));

	spheres[0] = sphere(Vec3f(1.281f, 0.371f, 2.079f), 0.5f);
	spheres[1] = sphere(Vec3f(1.829f, 1.031f, 2.155f), 0.5f);
	spheres[2] = sphere(Vec3f(2.303f, 0.928f, 0.609f), 0.5f);
	spheres[3] = sphere(Vec3f(-0.172f, 0.611f, 1.549f), 0.5f);

	capsules[0] = capsule(Vec3f(-0.756f, 0.722f, 1.434f), Vec3f(-92.316f, -64.865f, 1.434f), 0.25f, 0.5f);
	capsules[1] = capsule(Vec3f(-1.267f, 0.477f, 1.739f), Vec3f(-59.821f, -19.591f, 1.434f), 0.25f, 0.5f);
	capsules[2] = capsule(Vec3f(0.712f, 0.734f, 0.674f), Vec3f(-149.690f, 69.503f, -57.249f), 0.25f, 0.5f);
	// capsules[3] = capsule(Vec3f(-0.6f, 0.5f, 0.0f), 0.25f, 0.5f);
	
	
	auto satC2C = scn->addNode(std::make_shared<CollisionDetector<DataType3f>>());
	cubes[0]->connect(satC2C->importShapeB());
	cubes[1]->connect(satC2C->importShapeA());

	auto satS2S = scn->addNode(std::make_shared<CollisionDetector<DataType3f>>());
	spheres[0]->connect(satS2S->importShapeB());
	spheres[1]->connect(satS2S->importShapeA());

	auto satCa2Ca = scn->addNode(std::make_shared<CollisionDetector<DataType3f>>());
	capsules[0]->connect(satCa2Ca->importShapeB());
	capsules[1]->connect(satCa2Ca->importShapeA());

	auto satC2Ca = scn->addNode(std::make_shared<CollisionDetector<DataType3f>>());
	cubes[0]->connect(satC2Ca->importShapeB());
	capsules[2]->connect(satC2Ca->importShapeA());


	auto satC2S = scn->addNode(std::make_shared<CollisionDetector<DataType3f>>());
	cubes[0]->connect(satC2S->importShapeB());
	spheres[2]->connect(satC2S->importShapeA());

	auto satCa2S = scn->addNode(std::make_shared<CollisionDetector<DataType3f>>());
	capsules[0]->connect(satCa2S->importShapeB());
	spheres[3]->connect(satCa2S->importShapeA());

	auto satM2M = scn->addNode(std::make_shared<CollisionDetector<DataType3f>>());
	medalCones[0]->connect(satM2M->importShapeA());
	medalCones[1]->connect(satM2M->importShapeB());

	auto satMS2M = scn->addNode(std::make_shared<CollisionDetector<DataType3f>>());
	medalslabs[0]->connect(satMS2M->importShapeA());
	medalCones[1]->connect(satMS2M->importShapeB());

	auto satMS2MS = scn->addNode(std::make_shared<CollisionDetector<DataType3f>>());
	medalslabs[0]->connect(satMS2MS->importShapeA());
	medalslabs[1]->connect(satMS2MS->importShapeB());
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


