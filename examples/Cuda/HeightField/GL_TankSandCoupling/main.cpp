#include <GlfwApp.h>

#include <SceneGraph.h>

#include <HeightField/GranularMedia.h>

#include "Mapping/HeightFieldToTriangleSet.h"

#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>

#include <HeightField/SurfaceParticleTracking.h>
#include <HeightField/RigidSandCoupling.h>

#include "GltfLoader.h"
#include "RigidBody/ConfigurableBody.h"

#include "Collision/NeighborElementQuery.h"
#include "Collision/CollistionDetectionBoundingBox.h"

#include "RigidBody/Module/PJSConstraintSolver.h"

#include "RigidBody/Module/ContactsUnion.h"

#include <Module/GLPhotorealisticInstanceRender.h>
#include "BasicShapes/PlaneModel.h"
#include "RigidBody/MultibodySystem.h"

using namespace std;
using namespace dyno;

std::shared_ptr<ConfigurableBody<DataType3f>> getTank(std::shared_ptr<SceneGraph> scn)
{
	auto vehicle = scn->addNode(std::make_shared<ConfigurableBody<DataType3f>>());

	auto gltf = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	gltf->varFileName()->setValue(getAssetPath() + "gltf/Tank/Tank.gltf");
	gltf->setVisible(false);

	gltf->stateTextureMesh()->connect(vehicle->inTextureMesh());

	VehicleBind configData;

	Vec3f angle = Vec3f(0, 0, 90);
	Quat<Real> q = Quat<Real>(angle[2] * M_PI / 180, angle[1] * M_PI / 180, angle[0] * M_PI / 180);
	;
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("Main", 0), 0, Box));//
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("Head", 1), 1, Box));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("L0", 2), 2, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("L1", 3), 3, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("L2", 4), 4, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("L3", 5), 5, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("L4", 6), 6, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("L5", 7), 7, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("L6", 8), 8, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("R0", 9), 9, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("R1", 10), 10, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("R2", 11), 11, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("R3", 12), 12, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("R4", 13), 13, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("R5", 14), 14, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("R6", 15), 15, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));

	for (size_t i = 0; i < configData.mVehicleRigidBodyInfo.size(); i++)
	{
		configData.mVehicleRigidBodyInfo[i].capsuleLength = 0.3;
	}
	float speed = 5.5;

	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("L0", 2), Name_Shape("Main", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("L1", 3), Name_Shape("Main", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("L2", 4), Name_Shape("Main", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("L3", 5), Name_Shape("Main", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("L4", 6), Name_Shape("Main", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("L5", 7), Name_Shape("Main", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("L6", 8), Name_Shape("Main", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));

	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("R0", 9), Name_Shape("Main", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("R1", 10), Name_Shape("Main", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("R2", 11), Name_Shape("Main", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("R3", 12), Name_Shape("Main", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("R4", 13), Name_Shape("Main", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("R5", 14), Name_Shape("Main", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("R6", 15), Name_Shape("Main", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));

	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("Head", 1), Name_Shape("Main", 0), Fixed, Vec3f(1, 0, 0), Vec3f(0), false, 0));

	vehicle->varVehicleConfiguration()->setValue(configData);

	vehicle->varRotation()->setValue(Vec3f(0, 0, 0));

	std::vector<Transform3f> vehicleTransforms;

	vehicleTransforms.push_back(Transform3f(Vec3f(1, 0, 0), Quat1f(0, Vec3f(0, 1, 0)).toMatrix3x3()));

	vehicle->varVehiclesTransform()->setValue(vehicleTransforms);

	return vehicle;
}

std::shared_ptr<ConfigurableBody<DataType3f>> getVehicle(std::shared_ptr<SceneGraph> scn)
{
	auto vehicle = scn->addNode(std::make_shared<ConfigurableBody<DataType3f>>());

	auto gltf = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	gltf->varFileName()->setValue(getAssetPath() + "gltf/Aurora950SX/Aurora950SX.gltf");
	gltf->setVisible(false);

	gltf->stateTextureMesh()->connect(vehicle->inTextureMesh());

	VehicleBind configData;

	Vec3f angle = Vec3f(0, 0, 90);
	Quat<Real> q = Quat<Real>(angle[2] * M_PI / 180, angle[1] * M_PI / 180, angle[0] * M_PI / 180);
	;
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("Body", 0), 0, Box));//
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("L1", 1), 1, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("L2", 2), 2, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("L3", 3), 3, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("L4", 4), 4, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("R1", 5), 5, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("R2", 6), 6, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("R3", 7), 7, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("R4", 8), 8, Capsule, Transform3f(Vec3f(0), q.toMatrix3x3(), Vec3f(1))));

	for (size_t i = 0; i < configData.mVehicleRigidBodyInfo.size(); i++)
	{
		configData.mVehicleRigidBodyInfo[i].capsuleLength = 0.3;
	}
	float speed = 5.5;

	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("L1", 1), Name_Shape("Body", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("L2", 2), Name_Shape("Body", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("L3", 3), Name_Shape("Body", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("L4", 4), Name_Shape("Body", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));

	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("R1", 5), Name_Shape("Main", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("R2", 6), Name_Shape("Main", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("R3", 7), Name_Shape("Main", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("R4", 8), Name_Shape("Main", 0), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));

	vehicle->varVehicleConfiguration()->setValue(configData);
	vehicle->varRotation()->setValue(Vec3f(0, 0, 0));

	std::vector<Transform3f> vehicleTransforms;

	vehicleTransforms.push_back(Transform3f(Vec3f(1, 0, 0), Quat1f(0, Vec3f(0, 1, 0)).toMatrix3x3()));
	vehicle->varVehiclesTransform()->setValue(vehicleTransforms);

	return vehicle;

}


std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());

	auto tank = getVehicle(scn);
	plane->stateTriangleSet()->connect(tank->inTriangleSet());
	plane->setVisible(false);
	plane->varLengthX()->setValue(500);
	plane->varLengthZ()->setValue(500);


	auto multibody = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
	tank->connect(multibody->importVehicles());
	plane->stateTriangleSet()->connect(multibody->inTriangleSet());


	float spacing = 0.1f;
	uint res = 1024;
	auto sand = scn->addNode(std::make_shared<GranularMedia<DataType3f>>());
	sand->varOrigin()->setValue(-0.5f * Vec3f(res * spacing, 0.0f, res * spacing));
	sand->varSpacing()->setValue(spacing);
	sand->varWidth()->setValue(res);
	sand->varHeight()->setValue(res);
	sand->varDepth()->setValue(0.2);
	sand->varDepthOfDiluteLayer()->setValue(0.1);

	std::cout << "***************** Sand Resolution: " << res << ", " << res << "*********************"<<std::endl;


	auto mapper = std::make_shared<HeightFieldToTriangleSet<DataType3f>>();
	sand->stateHeightField()->connect(mapper->inHeightField());
	sand->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Color(0.8, 0.8, 0.8));
	sRender->varUseVertexNormal()->setValue(true);
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	sand->graphicsPipeline()->pushModule(sRender);

// 	auto tracking = scn->addNode(std::make_shared<SurfaceParticleTracking<DataType3f>>());
// 
// 	auto ptRender = tracking->graphicsPipeline()->findFirstModule<GLPointVisualModule>();
// 	ptRender->varPointSize()->setValue(0.01);
// 
// 	sand->connect(tracking->importGranularMedia());

	auto coupling = scn->addNode(std::make_shared<RigidSandCoupling<DataType3f>>());
	tank->connect(coupling->importRigidBodySystem());
	sand->connect(coupling->importGranularMedia());

	return scn;
}

int main()
{
	GlfwApp app;
	app.initialize(1024, 768);

	app.setSceneGraph(createScene());
	app.renderWindow()->getCamera()->setUnitScale(5);

	app.mainLoop();

	return 0;
}