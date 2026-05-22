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
	
	MultiBodyTuple configData;

	Vec3f angle = Vec3f(0, 0, 90);
	Quat<Real> q = Quat<Real>(angle[2] * M_PI / 180, angle[1] * M_PI / 180, angle[0] * M_PI / 180);
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("Main", 0, 0, RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("Head", 1, 1, RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("L0", 2, 2, RigidShapeType::SHAPE_CAPSULE, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("L1", 3, 3, RigidShapeType::SHAPE_CAPSULE, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("L2", 4, 4, RigidShapeType::SHAPE_CAPSULE, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("L3", 5, 5, RigidShapeType::SHAPE_CAPSULE, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("L4", 6, 6, RigidShapeType::SHAPE_CAPSULE, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("L5", 7, 7, RigidShapeType::SHAPE_CAPSULE, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("L6", 8, 8, RigidShapeType::SHAPE_CAPSULE, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("R0", 9, 9, RigidShapeType::SHAPE_CAPSULE, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("R1", 10, 10, RigidShapeType::SHAPE_CAPSULE, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("R2", 11, 11, RigidShapeType::SHAPE_CAPSULE, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("R3", 12, 12, RigidShapeType::SHAPE_CAPSULE, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("R4", 13, 13, RigidShapeType::SHAPE_CAPSULE, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("R5", 14, 14, RigidShapeType::SHAPE_CAPSULE, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("R6", 15, 15, RigidShapeType::SHAPE_CAPSULE, 100));

	int index = 0;

	for (auto it = configData.varRigidBodyConfigs()->begin();
		it != configData.varRigidBodyConfigs()->end();
		++it)
	{
		if (index == 0 ||index == 1)
			continue;
		auto* rigidPtr = dynamic_cast<TFTuple<RigidBodyTuple>*>((*it).get());

		auto rigid = configData.varRigidBodyConfigs()->getElement(it);
		auto base_ptr = (*rigid.varShapeConfigs()->begin()).get();
		auto* shapePtr = dynamic_cast<TFTuple<ShapeTuple>*>(base_ptr);

		auto shape = shapePtr->getValue();
		shape.varCapsuleLength()->setValue(0.2);
		shape.varRot()->setValue(q);
		shapePtr->setValue(shape);

		rigidPtr->setValue(rigid);
	}

	float speed = 5.5;
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("L0", 2, "Main", 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("L1", 3, "Main", 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("L2", 4, "Main", 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("L3", 5, "Main", 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("L4", 6, "Main", 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("L5", 7, "Main", 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("L6", 7, "Main", 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("R0", 9, "Main", 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("R1", 10, "Main", 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("R2", 11, "Main", 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("R3", 12, "Main", 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("R4", 13, "Main", 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("R5", 14, "Main", 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("R6", 15, "Main", 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("Head", 1, "Main", 0, JointType::JOINT_Fixed, Vec3f(1, 0, 0), Vec3f(0), true, 0));


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

	MultiBodyTuple configData;

	Vec3f angle = Vec3f(0, 0, 90);
	Quat<Real> q = Quat<Real>(angle[2] * M_PI / 180, angle[1] * M_PI / 180, angle[0] * M_PI / 180);
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("Body", 0, 0, RigidShapeType::SHAPE_BOX, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("L1", 1, 1, RigidShapeType::SHAPE_CAPSULE, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("L2", 2, 2, RigidShapeType::SHAPE_CAPSULE, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("L3", 3, 3, RigidShapeType::SHAPE_CAPSULE, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("L4", 4, 4, RigidShapeType::SHAPE_CAPSULE, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("R1", 5, 5, RigidShapeType::SHAPE_CAPSULE, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("R2", 6, 6, RigidShapeType::SHAPE_CAPSULE, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("R3", 7, 7, RigidShapeType::SHAPE_CAPSULE, 100));
	configData.varRigidBodyConfigs()->pushBack(RigidBodyTuple("R4", 8, 8, RigidShapeType::SHAPE_CAPSULE, 100));

	int index = 0;
	for (auto it = configData.varRigidBodyConfigs()->begin();
		it != configData.varRigidBodyConfigs()->end();
		++it, ++index)
	{
		if (index == 0)
			continue;
		auto* rigidPtr = dynamic_cast<TFTuple<RigidBodyTuple>*>((*it).get());

		auto rigid = configData.varRigidBodyConfigs()->getElement(it);
		auto base_ptr = (*rigid.varShapeConfigs()->begin()).get();
		auto* shapePtr = dynamic_cast<TFTuple<ShapeTuple>*>(base_ptr);

		auto shape = shapePtr->getValue();
		shape.varCapsuleLength()->setValue(0.2);
		shape.varRot()->setValue(q);
		shapePtr->setValue(shape);

		rigidPtr->setValue(rigid);

	}
	float speed = 5.5;

	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("L1", 1, "Body", 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("L2", 2, "Body", 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("L3", 3, "Body", 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("L4", 4, "Body", 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("R1", 5, "Body", 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("R2", 6, "Body", 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("R3", 7, "Body", 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));
	configData.varJointConfigs()->pushBack(MultiBodyJointTuple("R4", 8, "Body", 0, JointType::JOINT_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, speed));


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