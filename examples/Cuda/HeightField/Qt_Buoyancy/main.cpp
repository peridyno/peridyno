/**
 * Copyright 2022 Xiaowei He
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

#include <HeightField/Ocean.h>
#include <HeightField/OceanPatch.h>
#include <HeightField/Coupling.h>

#include "Module/ComputeModule.h"

#include <RigidBody/RigidMesh.h>

#include "Mapping/HeightFieldToTriangleSet.h"

#include <GLRenderEngine.h>
#include <GLSurfaceVisualModule.h>

using namespace std;
using namespace dyno;

/**
 * @brief An example to demonstrate the coupling between a boat and the ocean
 */

template<typename TDataType>
class DragBoat : public ComputeModule
{
public:
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	typedef typename TDataType::Matrix Matrix;

	DragBoat() {};
	virtual ~DragBoat() {};

	DEF_VAR_IN(Coord, Velocity, "Velocity");

	DEF_VAR_IN(Quat<Real>, Quaternion, "Rotation");

protected:
	void compute() override
	{
		auto quat = this->inQuaternion()->getData();

		Coord vel = this->inVelocity()->getData();

		Matrix rot = quat.toMatrix3x3();

		Coord vel_prime = rot.transpose() * vel;

		vel_prime[2] = 1.0;

		this->inVelocity()->setValue(rot * vel_prime);
	}
};

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto ocean = scn->addNode(std::make_shared<Ocean<DataType3f>>());

	auto patch = scn->addNode(std::make_shared<OceanPatch<DataType3f>>());
	patch->varWindType()->setValue(3);
	patch->connect(ocean->importOceanPatch());

	auto mapper = std::make_shared<HeightFieldToTriangleSet<DataType3f>>();

	ocean->stateHeightField()->connect(mapper->inHeightField());
	ocean->graphicsPipeline()->pushModule(mapper);


	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Vec3f(0, 0.2, 1.0));
	sRender->varUseVertexNormal()->setValue(true);
	sRender->varAlpha()->setValue(0.6);
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	ocean->graphicsPipeline()->pushModule(sRender);

	auto boat = scn->addNode(std::make_shared<RigidMesh<DataType3f>>());
	boat->varScale()->setValue(Vec3f(5));
	boat->varDensity()->setValue(150.0f);
	boat->stateVelocity()->setValue(Vec3f(10, 0, 0));
	boat->varEnvelopeName()->setValue(getAssetPath() + "obj/boat_boundary.obj");
	boat->varMeshName()->setValue(getAssetPath() + "obj/boat_mesh.obj");

	auto dragging = std::make_shared<DragBoat<DataType3f>>();
	boat->stateVelocity()->connect(dragging->inVelocity());
	boat->stateQuaternion()->connect(dragging->inQuaternion());
	boat->animationPipeline()->pushModule(dragging);

	auto rigidMeshRender = std::make_shared<GLSurfaceVisualModule>();
	rigidMeshRender->setColor(Vec3f(0.8, 0.8, 0.8));
	boat->stateMesh()->promoteOuput()->connect(rigidMeshRender->inTriangleSet());
	boat->graphicsPipeline()->pushModule(rigidMeshRender);

	
	auto coupling = scn->addNode(std::make_shared<Coupling<DataType3f>>());
	boat->connect(coupling->importRigidMesh());
	ocean->connect(coupling->importOcean());
	
	return scn;
}

int main()
{
	QtApp app;

	app.setSceneGraph(createScene());

	app.initialize(1024, 768);
	
	//Set the distance unit for the camera, the fault unit is meter
	app.renderWindow()->getCamera()->setUnitScale(50.0);

	app.mainLoop();

	return 0;
}
