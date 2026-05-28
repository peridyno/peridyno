#include <QtApp.h>
#include <GlfwGUI/GlfwApp.h>

#include <SceneGraph.h>

#include <BasicShapes/PlaneModel.h>

#include "GltfLoader.h"
#include "BasicShapes/PlaneModel.h"
#include "RigidBody/MultibodySystem.h"

#include  "RigidBody/J15.h"

using namespace dyno;


std::shared_ptr<SceneGraph> createJ15ControlScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto j15 = scn->addNode(std::make_shared<J15<DataType3f>>());

	auto multisystem = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
	multisystem->varGravityEnabled()->setValue(true);
	multisystem->varFrictionEnabled()->setValue(true);
	multisystem->varGravityValue()->setValue(9.8);
	
	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	j15->connect(multisystem->importVehicles());
	plane->stateTriangleSet()->connect(multisystem->inTriangleSet());
	plane->varLengthX()->setValue(220);
	plane->varLengthZ()->setValue(120);
	plane->varLocation()->setValue(Vec3f(0, -0.850,0));
	plane->varRotation()->setValue(Vec3f(0, 0, 0));

	return scn;
}

int main()
{
	QtApp app;
	//GlfwApp app;
	app.setSceneGraph(createJ15ControlScene());
	app.initialize(1280, 768);

	//Set the distance unit for the camera, the fault unit is meter
	//app.renderWindow()->getCamera()->setUnitScale(3.0f);

	app.mainLoop();

	return 0;
}
