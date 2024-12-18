#include <QtApp.h>
#include <SceneGraph.h>
#include <RigidBody/Vechicle.h>
#include <BasicShapes/PlaneModel.h>
#include "GltfLoader.h"
#include "BasicShapes/PlaneModel.h"
#include "MergeTextureMesh.h"
#include "ExtractShape.h"

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> creatCar()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto configCar = scn->addNode(std::make_shared<ConfigurableVehicle<DataType3f>>());

	auto jeep = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	jeep->varFileName()->setValue(getAssetPath() + "Jeep/JeepGltf/jeep.gltf");
	jeep->setVisible(false);

	auto extractBody = scn->addNode(std::make_shared<ExtractShape<DataType3f>>());
	extractBody->varShapeId()->setValue(std::vector<int>{4,5});
	extractBody->varShapeTransform()->setValue(std::vector<Transform3f>{Transform3f(),Transform3f()});
	jeep->stateTextureMesh()->connect(extractBody->inInTextureMesh());

	jeep->stateTextureMesh()->connect(configCar->inTextureMesh());

	auto race = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	race->varFileName()->setValue(getAssetPath() + "gltf/RaceGltf/Race_Mesh.gltf");
	race->setVisible(false);

	auto extractWheel = scn->addNode(std::make_shared<ExtractShape<DataType3f>>());
	extractWheel->varShapeId()->setValue(std::vector<int>{12,12,12,12});
	extractWheel->varOffset()->setValue(false);
	extractWheel->varShapeTransform()->setValue(
		std::vector<Transform3f>
	{
		Transform3f(Vec3f(0.8, 0.45, 1.73), Mat3f::identityMatrix(), Vec3f(1)),
			Transform3f(Vec3f(0.8, 0.45, -1.4), Mat3f::identityMatrix(), Vec3f(1)),
			Transform3f(Vec3f(-0.8, 0.45, 1.73), Quat1f(M_PI,Vec3f(0,1,0)).toMatrix3x3(), Vec3f(1)),
			Transform3f(Vec3f(-0.8, 0.45, -1.4), Quat1f(M_PI, Vec3f(0, 1, 0)).toMatrix3x3(), Vec3f(1)),
	}
	);
	race->stateTextureMesh()->connect(extractWheel->inInTextureMesh());

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
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("Block1", 6), 6, Box));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("Block2", 7), 7, Box));
	configData.mVehicleRigidBodyInfo.push_back(VehicleRigidBodyInfo(Name_Shape("Block3", 8), 8, Box));

	configData.mVehicleRigidBodyInfo[6].rigidGroup = 1;
	configData.mVehicleRigidBodyInfo[6].radius = 0.3;
	configData.mVehicleRigidBodyInfo[6].capsuleLength = 0.7;

	configData.mVehicleRigidBodyInfo[7].rigidGroup = 2;
	configData.mVehicleRigidBodyInfo[7].radius = 0.822;
	configData.mVehicleRigidBodyInfo[7].capsuleLength = 0.21;

	configData.mVehicleRigidBodyInfo[8].rigidGroup = 3;
	configData.mVehicleRigidBodyInfo[8].radius = 0.67;
	configData.mVehicleRigidBodyInfo[8].capsuleLength = 0.4;


	for (size_t i = 0; i < 4; i++)
	{
		configData.mVehicleRigidBodyInfo[i].capsuleLength = 0.3;
	}

	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("LF", 0), Name_Shape("Body", 5), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, 10));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("LB", 1), Name_Shape("Body", 5), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, 10));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("RF", 2), Name_Shape("Body", 5), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, 10));
	configData.mVehicleJointInfo.push_back(VehicleJointInfo(Name_Shape("RB", 3), Name_Shape("Body", 5), Hinge, Vec3f(1, 0, 0), Vec3f(0), true, 10));


	configCar->varVehicleConfiguration()->setValue(configData);

	configCar->varRotation()->setValue(Vec3f(0, 0, 0));

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->varLengthX()->setValue(50);
	plane->varLengthZ()->setValue(50);

	plane->stateTriangleSet()->connect(configCar->inTriangleSet());



	auto roadblock = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	roadblock->varFileName()->setValue(getAssetPath() + "gltf/Roadblock/Roadblock.gltf");
	roadblock->setVisible(false);

	auto extractBlock = scn->addNode(std::make_shared<ExtractShape<DataType3f>>());

	extractBlock->varShapeId()->setValue(std::vector<int>{0,2,1,4});
	extractBlock->varOffset()->setValue(false);
	extractBlock->varShapeTransform()->setValue(
		std::vector<Transform3f>
		{
			Transform3f(Vec3f(3, 0.95, 8), Mat3f::identityMatrix(), Vec3f(1)),
			Transform3f(Vec3f(1, 0.76, 8), Mat3f::identityMatrix(), Vec3f(1)),
			Transform3f(Vec3f(-1, 0.5, 8), Mat3f::identityMatrix(), Vec3f(1)),
			Transform3f(Vec3f(-3, 0.62, 8), Mat3f::identityMatrix(), Vec3f(1)),
		}
	);
	roadblock->stateTextureMesh()->connect(extractBlock->inInTextureMesh());


	auto merge = scn->addNode(std::make_shared<MergeTextureMesh<DataType3f>>());
	extractWheel->stateResult()->connect(merge->inFirst());
	extractBody->stateResult()->connect(merge->inSecond());

	auto mergeRoadblock = scn->addNode(std::make_shared<MergeTextureMesh<DataType3f>>());
	merge->stateTextureMesh()->connect(mergeRoadblock->inFirst());
	extractBlock->stateResult()->connect(mergeRoadblock->inSecond());

	extractBlock->setVisible(false);
	extractWheel->setVisible(false);
	extractBody->setVisible(false);
	mergeRoadblock->setVisible(false);
	merge->setVisible(false);

	mergeRoadblock->stateTextureMesh()->connect(configCar->inTextureMesh());

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


