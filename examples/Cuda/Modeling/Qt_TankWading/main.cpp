#include <QtApp.h>
#include <SceneGraph.h>
#include <GLRenderEngine.h>

#include <BasicShapes/CubeModel.h>

#include <Volume/BasicShapeToVolume.h>

#include <Multiphysics/VolumeBoundary.h>

#include <ParticleSystem/ParticleFluid.h>
#include <ParticleSystem/MakeParticleSystem.h>
#include <ParticleSystem/Emitters/SquareEmitter.h>
#include <ParticleSystem/Module/ParticleIntegrator.h>
#include <ParticleSystem/Module/IterativeDensitySolver.h>
#include <ParticleSystem/Module/ImplicitViscosity.h>

#include <Collision/NeighborPointQuery.h>

//Rendering
#include <GLSurfaceVisualModule.h>
#include <GLPhotorealisticInstanceRender.h>

#include <Commands/Merge.h>

#include <BasicShapes/CubeModel.h>
#include <Samplers/ShapeSampler.h>

#include <Node/GLPointVisualNode.h>

#include <SemiAnalyticalScheme/TriangularMeshBoundary.h>

#include <ColorMapping.h>
#include <Module/CalculateNorm.h>

#include <GltfLoader.h>

#include "Auxiliary/DataSource.h"

#include <RigidBody/Vehicle.h>
#include <RigidBody/ConfigurableBody.h>
#include <RigidBody/Module/InstanceTransform.h>

#include <Mapping/TextureMeshToTriangleSet.h>
#include <Mapping/MergeTriangleSet.h>
#include "ObjIO/ObjLoader.h"
#include "RigidBody/MultibodySystem.h"

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
		++it,index++)
	{
		if (index == 0 || index == 1)
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

	std::list<Transform3f> vehicleTransforms;

	vehicleTransforms.push_back(Transform3f(Vec3f(1, 0, 0), Quat1f(0, Vec3f(0, 1, 0)).toMatrix3x3()));

	vehicle->varVehiclesTransform()->assign(vehicleTransforms);

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

	std::list<Transform3f> vehicleTransforms;

	vehicleTransforms.push_back(Transform3f(Vec3f(1, 0, 0), Quat1f(0, Vec3f(0, 1, 0)).toMatrix3x3()));
	vehicle->varVehiclesTransform()->assign(vehicleTransforms);

	return vehicle;

}


std::shared_ptr<SceneGraph> creatScene();

float total_scale = 8;

std::shared_ptr<SceneGraph> creatScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setAsynchronousSimulation(true);

	//***************************************Scene Setting***************************************//
	// Scene Setting
	scn->setTotalTime(3.0f);
	scn->setGravity(Vec3f(0.0f, -9.8f, 0.0f));
	scn->setLowerBound(Vec3f(-0.5f, 0.0f, -4.0f) * total_scale);
	scn->setUpperBound(Vec3f(0.5f, 1.0f, 4.0f) * total_scale);


	// Create Var
	Vec3f velocity = Vec3f(0, 0, 6);
	Color color = Color(1, 1, 1);

	Vec3f LocationBody = Vec3f(0, 0.01, -1);

	Vec3f anglurVel = Vec3f(100, 0, 0);
	Vec3f scale = Vec3f(0.4, 0.4, 0.4);


	auto tank = getVehicle(scn);


	auto gltfRoad = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	gltfRoad->varFileName()->setValue(getAssetPath() + "gltf/Road_Gltf/Road_Tex.gltf");
	gltfRoad->varLocation()->setValue(Vec3f(0, 0, 3.488));

	auto roadMeshConverter = std::make_shared<TextureMeshToTriangleSet<DataType3f>>();
	gltfRoad->stateTextureMesh()->connect(roadMeshConverter->inTextureMesh());
	gltfRoad->animationPipeline()->pushModule(roadMeshConverter);

	auto tsJeep = gltfRoad->animationPipeline()->promoteOutputToNode(roadMeshConverter->outTriangleSet());

	auto transformer = std::make_shared<InstanceTransform<DataType3f>>();
	tank->stateCenter()->connect(transformer->inCenter());
	tank->stateRotationMatrix()->connect(transformer->inRotationMatrix());
	tank->stateBindingPair()->connect(transformer->inBindingPair());
	tank->stateBindingTag()->connect(transformer->inBindingTag());
	tank->stateInstanceTransform()->connect(transformer->inInstanceTransform());
	tank->animationPipeline()->pushModule(transformer);


	auto texMeshConverter = std::make_shared<TextureMeshToTriangleSet<DataType3f>>();
	tank->inTextureMesh()->connect(texMeshConverter->inTextureMesh());
	transformer->outInstanceTransform()->connect(texMeshConverter->inTransform());
	tank->animationPipeline()->pushModule(texMeshConverter);
	tank->varLocation()->setValue(Vec3f(-0.574, 0.364, -2.9));

	auto tsMerger = scn->addNode(std::make_shared<MergeTriangleSet<DataType3f>>());
	//texMeshConverter->outTriangleSet()->connect(tsMerger->inFirst());
	tank->animationPipeline()->promoteOutputToNode(texMeshConverter->outTriangleSet())->connect(tsMerger->inFirst());
	tsJeep->connect(tsMerger->inSecond());


	auto multibody = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
	tank->connect(multibody->importVehicles());
	tsJeep->connect(multibody->inTriangleSet());


	//*************************************** Cube Sample ***************************************//
	// Cube 
	auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube->varLocation()->setValue(Vec3f(0, 0.15, 3.436));
	cube->varLength()->setValue(Vec3f(2.1, 0.12, 16));
	cube->varScale()->setValue(Vec3f(2, 1, 0.932));
	cube->graphicsPipeline()->disable();

	auto cubeSmapler = scn->addNode(std::make_shared<ShapeSampler<DataType3f>>());
	cubeSmapler->varSamplingDistance()->setValue(0.004f * total_scale);
	cube->connect(cubeSmapler->importShape());
	cubeSmapler->graphicsPipeline()->disable();

	//MakeParticleSystem
	auto particleSystem = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());
	cubeSmapler->statePointSet()->promoteOuput()->connect(particleSystem->inPoints());

	//*************************************** Fluid ***************************************//
	//Particle fluid node
	auto fluid = scn->addNode(std::make_shared<ParticleFluid<DataType3f>>());
	fluid->setDt(0.004);
	{
		fluid->animationPipeline()->clear();

		auto smoothingLength = fluid->animationPipeline()->createModule<FloatingNumber<DataType3f>>();
		smoothingLength->varValue()->setValue(0.006f * total_scale);

		auto samplingDistance = fluid->animationPipeline()->createModule<FloatingNumber<DataType3f>>();
		samplingDistance->varValue()->setValue(Real(0.004) * total_scale);

		auto integrator = std::make_shared<ParticleIntegrator<DataType3f>>();
		fluid->stateTimeStep()->connect(integrator->inTimeStep());
		fluid->statePosition()->connect(integrator->inPosition());
		fluid->stateVelocity()->connect(integrator->inVelocity());
		fluid->animationPipeline()->pushModule(integrator);

		auto nbrQuery = std::make_shared<NeighborPointQuery<DataType3f>>();
		smoothingLength->outFloating()->connect(nbrQuery->inRadius());
		fluid->statePosition()->connect(nbrQuery->inPosition());
		fluid->animationPipeline()->pushModule(nbrQuery);

		auto density = std::make_shared<IterativeDensitySolver<DataType3f>>();
		density->varKappa()->setValue(0.01f);

		fluid->stateTimeStep()->connect(density->inTimeStep());
		fluid->statePosition()->connect(density->inPosition());
		fluid->stateVelocity()->connect(density->inVelocity());
		nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
		fluid->animationPipeline()->pushModule(density);

		smoothingLength->outFloating()->connect(density->inSmoothingLength());
		samplingDistance->outFloating()->connect(density->inSamplingDistance());

		auto viscosity = std::make_shared<ImplicitViscosity<DataType3f>>();
		viscosity->varViscosity()->setValue(Real(3.0));
		fluid->stateTimeStep()->connect(viscosity->inTimeStep());
		smoothingLength->outFloating()->connect(viscosity->inSmoothingLength());
		samplingDistance->outFloating()->connect(viscosity->inSamplingDistance());
		fluid->statePosition()->connect(viscosity->inPosition());
		fluid->stateVelocity()->connect(viscosity->inVelocity());
		nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
		fluid->animationPipeline()->pushModule(viscosity);
	}

	//Setup the point render
	{
		auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
		fluid->stateVelocity()->connect(calculateNorm->inVec());
		fluid->graphicsPipeline()->pushModule(calculateNorm);

		auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
		colorMapper->varMax()->setValue(2.0f);
		calculateNorm->outNorm()->connect(colorMapper->inScalar());
		fluid->graphicsPipeline()->pushModule(colorMapper);

		auto ptRender = std::make_shared<GLPointVisualModule>();
		ptRender->varBaseColor()->setValue(Color(1, 0, 0));
		ptRender->varPointSize()->setValue(0.01);
		ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);

		fluid->statePointSet()->connect(ptRender->inPointSet());
		colorMapper->outColor()->connect(ptRender->inColor());

		fluid->graphicsPipeline()->pushModule(ptRender);
	}

	particleSystem->connect(fluid->importInitialStates());

	//TriangularMeshBoundary
	auto meshBoundary = scn->addNode(std::make_shared<TriangularMeshBoundary<DataType3f>>());
	meshBoundary->varThickness()->setValue(0.005f * total_scale);

	fluid->connect(meshBoundary->importParticleSystems());
	tsMerger->stateTriangleSet()->connect(meshBoundary->inTriangleSet());

	//Create a boundary
	auto cubeBoundary = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cubeBoundary->varLocation()->setValue(Vec3f(0.0f, 3.006f, 3.476f));
	cubeBoundary->varScale()->setValue(Vec3f(1.0f, 1.0f, 0.875f));
	cubeBoundary->varLength()->setValue(Vec3f(9.2f, 6.0f, 19.200f));
	cubeBoundary->setVisible(false);

	auto cube2vol = scn->addNode(std::make_shared<BasicShapeToVolume<DataType3f>>());
	cube2vol->varGridSpacing()->setValue(0.1f);
	cube2vol->varInerted()->setValue(true);
	cubeBoundary->connect(cube2vol->importShape());

	auto container = scn->addNode(std::make_shared<VolumeBoundary<DataType3f>>());
	cube2vol->connect(container->importVolumes());

	fluid->connect(container->importParticleSystems());


	return scn;
}

int main()
{
	QtApp window;
	window.setSceneGraph(creatScene());
	window.initialize(1366, 768);

	//Set the distance unit for the camera, the fault unit is meter
	window.renderWindow()->getCamera()->setUnitScale(3.0f);

	auto renderer = std::dynamic_pointer_cast<dyno::GLRenderEngine>(window.renderWindow()->getRenderEngine());
	if (renderer) {
		renderer->setEnvStyle(EEnvStyle::Studio);
		renderer->showGround = false;
		renderer->setUseEnvmapBackground(false);

	}

	window.mainLoop();

	return 0;
}