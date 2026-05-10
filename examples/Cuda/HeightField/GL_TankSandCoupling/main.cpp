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

	MultiBodyBind configData;

	Vec3f angle = Vec3f(0, 0, 90);
	Quat<Real> q = Quat<Real>(angle[2] * M_PI / 180, angle[1] * M_PI / 180, angle[0] * M_PI / 180);
	;
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("Main", 0), 0, ConfigShapeType::CONFIG_BOX));//
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("Head", 1), 1, ConfigShapeType::CONFIG_BOX));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("L0", 2), 2, ConfigShapeType::CONFIG_CAPSULE));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("L1", 3), 3, ConfigShapeType::CONFIG_CAPSULE));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("L2", 4), 4, ConfigShapeType::CONFIG_CAPSULE));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("L3", 5), 5, ConfigShapeType::CONFIG_CAPSULE));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("L4", 6), 6, ConfigShapeType::CONFIG_CAPSULE ));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("L5", 7), 7, ConfigShapeType::CONFIG_CAPSULE ));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("L6", 8), 8, ConfigShapeType::CONFIG_CAPSULE ));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("R0", 9), 9, ConfigShapeType::CONFIG_CAPSULE ));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("R1", 10), 10, ConfigShapeType::CONFIG_CAPSULE ));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("R2", 11), 11, ConfigShapeType::CONFIG_CAPSULE ));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("R3", 12), 12, ConfigShapeType::CONFIG_CAPSULE ));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("R4", 13), 13, ConfigShapeType::CONFIG_CAPSULE ));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("R5", 14), 14, ConfigShapeType::CONFIG_CAPSULE ));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("R6", 15), 15, ConfigShapeType::CONFIG_CAPSULE ));

	for (size_t i = 0; i < configData.rigidBodyConfigs.size(); i++)
	{
		configData.rigidBodyConfigs[i].shapeConfigs[0].capsuleLength = 0.3;
		configData.rigidBodyConfigs[i].shapeConfigs[0].rot = q;
	}
	float speed = 5.5;

	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("L0", 2), NameRigidID("Main", 0), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("L1", 3), NameRigidID("Main", 0), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("L2", 4), NameRigidID("Main", 0), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("L3", 5), NameRigidID("Main", 0), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("L4", 6), NameRigidID("Main", 0), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("L5", 7), NameRigidID("Main", 0), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("L6", 8), NameRigidID("Main", 0), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));

	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("R0", 9), NameRigidID("Main", 0), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("R1", 10), NameRigidID("Main", 0), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("R2", 11), NameRigidID("Main", 0), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("R3", 12), NameRigidID("Main", 0), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("R4", 13), NameRigidID("Main", 0), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("R5", 14), NameRigidID("Main", 0), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("R6", 15), NameRigidID("Main", 0), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));

	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("Head", 1), NameRigidID("Main", 0), ConfigJointType::CONFIG_Fixed, Vec3f(1, 0, 0), Vec3f(0), false, 0));

	vehicle->varConfiguration()->setValue(configData);

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

	MultiBodyBind configData;

	Vec3f angle = Vec3f(0, 0, 90);
	Quat<Real> q = Quat<Real>(angle[2] * M_PI / 180, angle[1] * M_PI / 180, angle[0] * M_PI / 180);
	;
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("Body", 0), 0, ConfigShapeType::CONFIG_BOX));//
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("L1", 1), 1, ConfigShapeType::CONFIG_CAPSULE ));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("L2", 2), 2, ConfigShapeType::CONFIG_CAPSULE ));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("L3", 3), 3, ConfigShapeType::CONFIG_CAPSULE ));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("L4", 4), 4, ConfigShapeType::CONFIG_CAPSULE ));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("R1", 5), 5, ConfigShapeType::CONFIG_CAPSULE ));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("R2", 6), 6, ConfigShapeType::CONFIG_CAPSULE ));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("R3", 7), 7, ConfigShapeType::CONFIG_CAPSULE ));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("R4", 8), 8, ConfigShapeType::CONFIG_CAPSULE ));

	for (size_t i = 0; i < configData.rigidBodyConfigs.size(); i++)
	{
		configData.rigidBodyConfigs[i].shapeConfigs[0].capsuleLength = 0.3;
		configData.rigidBodyConfigs[i].shapeConfigs[0].rot = q;
	}
	float speed = 5.5;

	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("L1", 1), NameRigidID("Body", 0), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("L2", 2), NameRigidID("Body", 0), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("L3", 3), NameRigidID("Body", 0), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("L4", 4), NameRigidID("Body", 0), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));

	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("R1", 5), NameRigidID("Main", 0), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("R2", 6), NameRigidID("Main", 0), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("R3", 7), NameRigidID("Main", 0), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("R4", 8), NameRigidID("Main", 0), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));

	vehicle->varConfiguration()->setValue(configData);
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
	sRender->varBaseColor()->setValue(Color(0.8, 0.8, 0.8));
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