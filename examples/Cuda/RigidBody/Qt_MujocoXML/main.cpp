#include <QtApp.h>

#include <SceneGraph.h>
#include <HeightField/GranularMedia.h>
#include <BasicShapes/PlaneModel.h>
#include <BasicShapes/PlaneModel.h>

#include <RigidBody/ConfigurableBody.h>
#include <RigidBody/Module/CarDriver.h>

#include "BasicShapes/PlaneModel.h"
#include "FBXLoader/FBXLoader.h"
#include "RigidBody/Module/AnimationDriver.h"
#include "RigidBody/MultibodySystem.h"
#include <HeightField/SurfaceParticleTracking.h>
#include <HeightField/RigidSandCoupling.h>
#include "MujocoLoader/MujocoXMLLoader.h"
#include "Commands/ExtractShape.h"
#include "RigidBody/MultibodySystem.h"
#include "BasicShapes/PlaneModel.h"

using namespace std;
using namespace dyno;


std::shared_ptr<SceneGraph> creatCar()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	
	auto mujocoB2 = scn->addNode(std::make_shared<MujocoXMLLoader<DataType3f>>());
	mujocoB2->varFilePath()->setValue(getAssetPath() + "Mujoco/b2/b2.xml");
	mujocoB2->varLocation()->setValue(Vec3f(0,0.1,0));

	auto mujocoGo2 = scn->addNode(std::make_shared<MujocoXMLLoader<DataType3f>>());
	mujocoGo2->varFilePath()->setValue(getAssetPath() + "Mujoco/go2/go2.xml");
	mujocoGo2->varLocation()->setValue(Vec3f(0.5, 0.1, 0));

	auto multi = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
	mujocoB2->connect(multi->importVehicles());
	mujocoGo2->connect(multi->importVehicles());
	

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->stateTriangleSet()->connect(multi->inTriangleSet());
	plane->varScale()->setValue(Vec3f(20));
	return scn;
}

int main()
{
	QtApp app;
	app.setSceneGraph(creatCar());
	app.initialize(1280, 768);

	//Set the distance unit for the camera, the fault unit is meter

	app.mainLoop();

	return 0;
}


