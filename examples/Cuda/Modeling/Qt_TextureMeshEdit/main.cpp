#include <QtApp.h>
#include <SceneGraph.h>

#include <RigidBody/ConfigurableBody.h>
#include <BasicShapes/PlaneModel.h>

#include "GltfLoader.h"
#include "BasicShapes/PlaneModel.h"
#include "Commands/TextureMeshMerge.h"
#include "Commands/ExtractShape.h"
#include "RigidBody/MultibodySystem.h"
#include <GLRenderEngine.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> creatCar()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();



	auto jeep = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	jeep->varFileName()->setValue(getAssetPath() + "Jeep/JeepGltf/jeep.gltf");
	jeep->setVisible(false);

	auto extractBody = scn->addNode(std::make_shared<ExtractShape<DataType3f>>());
	extractBody->varShapeId()->setValue(std::vector<int>{4,5});
	extractBody->varShapeTransform()->setValue(std::vector<Transform3f>{Transform3f(),Transform3f()});
	jeep->stateTextureMesh()->connect(extractBody->inInTextureMesh());

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


	auto configCar = scn->addNode(std::make_shared<ConfigurableBody<DataType3f>>());
	MultiBodyBind configData;

	Vec3f angle = Vec3f(0, 0, 90);
	Quat<Real> q = Quat<Real>(angle[2] * M_PI / 180, angle[1] * M_PI / 180, angle[0] * M_PI / 180);
	;
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("LF", 0), 0, ConfigShapeType::CONFIG_CAPSULE));//
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("LB", 1), 1, ConfigShapeType::CONFIG_CAPSULE));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("RF", 2), 2, ConfigShapeType::CONFIG_CAPSULE));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("RB", 3), 3, ConfigShapeType::CONFIG_CAPSULE));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("BackWheel", 4), 4, ConfigShapeType::CONFIG_BOX));
	configData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("Body", 5), 5, ConfigShapeType::CONFIG_BOX));


	for (size_t i = 0; i < 4; i++)
	{
		configData.rigidBodyConfigs[i].shapeConfigs[0].capsuleLength = 0.3;
		configData.rigidBodyConfigs[i].shapeConfigs[0].rot = q;
	}

	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("LF", 0), NameRigidID("Body", 5), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, 10));
	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("LB", 1), NameRigidID("Body", 5), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, 10));
	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("RF", 2), NameRigidID("Body", 5), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, 10));
	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("RB", 3), NameRigidID("Body", 5), ConfigJointType::CONFIG_Hinge, Vec3f(1, 0, 0), Vec3f(0), true, 10));
	configData.jointConfigs.push_back(MultiBodyJointConfig(NameRigidID("BackWheel", 4), NameRigidID("Body", 5), ConfigJointType::CONFIG_Fixed, Vec3f(1, 0, 0), Vec3f(0), true, 0));


	configCar->varConfiguration()->setValue(configData);

	configCar->varRotation()->setValue(Vec3f(0, 0, 0));


	//MergeTextureShape
	auto merge = scn->addNode(std::make_shared<TextureMeshMerge<DataType3f>>());
	extractWheel->stateResult()->connect(merge->inFirst());
	extractBody->stateResult()->connect(merge->inSecond());
	merge->stateTextureMesh()->connect(configCar->inTextureMesh());

	//TextureMesh Import
	auto roadblock = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	roadblock->varFileName()->setValue(getAssetPath() + "gltf/Roadblock/Roadblock.gltf");
	roadblock->setVisible(false);

	//TextureMesh Edit - Extract
	auto extractBlock = scn->addNode(std::make_shared<ExtractShape<DataType3f>>());

	extractBlock->varShapeId()->setValue(std::vector<int>{0,2,1,4});
	extractBlock->varOffset()->setValue(false);
	//Transform Selected Shape
	extractBlock->varShapeTransform()->setValue(
		std::vector<Transform3f>
		{
			Transform3f(Vec3f(0, 0.95, 11.8), Mat3f::identityMatrix(), Vec3f(1)),
			Transform3f(Vec3f(1, 0.76, 8), Mat3f::identityMatrix(), Vec3f(1)),
			Transform3f(Vec3f(-1, 0.5, 8), Mat3f::identityMatrix(), Vec3f(1)),
			Transform3f(Vec3f(0, 0.62, 10), Mat3f::identityMatrix(), Vec3f(1)),
		}
	);
	roadblock->stateTextureMesh()->connect(extractBlock->inInTextureMesh());



	extractBlock->setVisible(false);
	extractWheel->setVisible(false);
	extractBody->setVisible(false);
	merge->setVisible(false);


	auto configBlock = scn->addNode(std::make_shared<ConfigurableBody<DataType3f>>());

	MultiBodyBind configBlockData;
	configBlockData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("Block1", 0), 0, ConfigShapeType::CONFIG_BOX));
	configBlockData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("Block2", 1), 1, ConfigShapeType::CONFIG_BOX));
	configBlockData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("Block3", 2), 2, ConfigShapeType::CONFIG_BOX));
	configBlockData.rigidBodyConfigs.push_back(RigidBodyConfig(NameRigidID("Block4", 3), 3, ConfigShapeType::CONFIG_BOX));

	configBlockData.rigidBodyConfigs[0].ConfigGroup = 1;
	configBlockData.rigidBodyConfigs[1].ConfigGroup = 2;
	configBlockData.rigidBodyConfigs[2].ConfigGroup = 3;

	configBlock->varConfiguration()->setValue(configBlockData);

	extractBlock->stateResult()->connect(configBlock->inTextureMesh());

	auto multibody = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
	configCar->connect(multibody->importVehicles());
	configBlock->connect(multibody->importVehicles());

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->varLengthX()->setValue(60);
	plane->varLengthZ()->setValue(60);
	plane->varSegmentX()->setValue(10);
	plane->varSegmentZ()->setValue(10);

	plane->stateTriangleSet()->connect(multibody->inTriangleSet());


	return scn;
}

int main()
{
	QtApp app;
	app.setSceneGraph(creatCar());
	app.initialize(1280, 768);

	//Set the distance unit for the camera, the fault unit is meter
	app.renderWindow()->getCamera()->setUnitScale(3.0f);

	auto renderer = std::dynamic_pointer_cast<dyno::GLRenderEngine>(app.renderWindow()->getRenderEngine());
	if (renderer) {
		renderer->setEnvStyle(EEnvStyle::Studio);
		renderer->showGround = false;
		renderer->setUseEnvmapBackground(false);

	}


	app.mainLoop();

	return 0;
}


