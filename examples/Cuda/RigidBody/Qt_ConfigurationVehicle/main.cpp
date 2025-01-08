#include <QtApp.h>

#include <SceneGraph.h>

#include <RigidBody/ConfigurableBody.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Mapping/ContactsToEdgeSet.h>
#include <Mapping/ContactsToPointSet.h>
#include <Mapping/AnchorPointToPointSet.h>

#include "Collision/NeighborElementQuery.h"
#include "Collision/CollistionDetectionTriangleSet.h"
#include "Collision/CollistionDetectionBoundingBox.h"

#include <Module/GLPhotorealisticInstanceRender.h>

#include <BasicShapes/PlaneModel.h>

#include "GltfLoader.h"

#include "GltfLoader.h"
#include "BasicShapes/PlaneModel.h"
#include "RigidBody/MultibodySystem.h"


using namespace std;
using namespace dyno;


std::shared_ptr<SceneGraph> creatCar()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto configCar = scn->addNode(std::make_shared<ConfigurableBody<DataType3f>>());

	auto gltf = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	gltf->varFileName()->setValue(getAssetPath() + "Jeep/JeepGltf/jeep.gltf");
	gltf->setVisible(false);

	gltf->stateTextureMesh()->connect(configCar->inTextureMesh());


	VehicleBind configData;

	Vec3f angle = Vec3f(0, 0, 90);
	Quat<Real> q = Quat<Real>(angle[2] * M_PI / 180, angle[1] * M_PI / 180, angle[0] * M_PI / 180);
	;
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("LF", 0), 0, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));//
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("LB", 1), 1, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("RF", 2), 2, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("RB", 3), 3, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("BackWheel", 4), 4, Box));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("Body", 5), 5, Box));

	for (size_t i = 0; i < 4; i++)
	{
		configData.mVehicleRigidBodyInfo[i].capsuleLength = 0.3;
	}

	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("LF", 0), Name_Shape("Body", 5), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, 10));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("LB", 1), Name_Shape("Body", 5), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, 10));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("RF", 2), Name_Shape("Body", 5), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, 10));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("RB", 3), Name_Shape("Body", 5), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, 10));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("BackWheel", 4), Name_Shape("Body", 5), Fixed, Vec3f(1, 0, 0), Vec3f(0), true, 0));


	configCar->varVehicleConfiguration()->setValue(configData);

	configCar->varRotation()->setValue(Vec3f(0, 45, 0));



	std::vector<Transform3f> vehicleTransforms;

	vehicleTransforms.push_back(Transform3f(Vec3f(-1,0,0), Quat1f(0, Vec3f(0, 1, 0)).toMatrix3x3()));
	vehicleTransforms.push_back(Transform3f(Vec3f(5, 0.5, -1), Quat1f(M_PI, Vec3f(0, 1, 0)).toMatrix3x3()));

	configCar->varVehiclesTransform()->setValue(vehicleTransforms);

	auto multibody = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
	configCar->connect(multibody->importVehicles());
	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->varLengthX()->setValue(50);
	plane->varLengthZ()->setValue(50);
	plane->varSegmentX()->setValue(5);
	plane->varSegmentZ()->setValue(5);

	plane->stateTriangleSet()->connect(configCar->inTriangleSet());
	plane->stateTriangleSet()->connect(multibody->inTriangleSet());

 	//auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
 	//configCar->stateTopology()->connect(mapper->inDiscreteElements());
 	//configCar->graphicsPipeline()->pushModule(mapper);
 
 	//auto sRender = std::make_shared<GLSurfaceVisualModule>();
 	//sRender->setColor(Color(0.3f, 0.5f, 0.9f));
 	//sRender->setAlpha(0.8f);
 	//sRender->setRoughness(0.7f);
 	//sRender->setMetallic(3.0f);
 	//mapper->outTriangleSet()->connect(sRender->inTriangleSet());
 	//configCar->graphicsPipeline()->pushModule(sRender);

	return scn;
}

int main()
{
	QtApp app;
	app.setSceneGraph(creatCar());
	app.initialize(1280, 768);

	//Set the distance unit for the camera, the fault unit is meter
	app.renderWindow()->getCamera()->setUnitScale(3.0f);

	app.mainLoop();

	return 0;
}
