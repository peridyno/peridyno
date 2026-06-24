#include "QtGUI/QtApp.h"

#include "SceneGraph.h"

#include <RigidBody/ArticulatedBody.h>
#include <RigidBody/MultibodySystem.h>
#include <RigidBody/Module/InstanceTransform.h>

#include <ParticleSystem/Emitters/CircularEmitter.h>
#include "ParticleSystem/ParticleFluid.h"

#include "SemiAnalyticalScheme/SemiAnalyticalParticleFluid.h"

#include "Mapping/MergeTriangleSet.h"

#include "Collision/NeighborPointQuery.h"

#include "Module/CalculateNorm.h"

#include <ColorMapping.h>

#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLInstanceVisualModule.h>
#include <GLWireframeVisualModule.h>

#include "ParticleSystem/MakeParticleSystem.h"

#include "initializeModeling.h"
#include "ParticleSystem/initializeParticleSystem.h"
#include "SemiAnalyticalScheme/initializeSemiAnalyticalScheme.h"
#include "SemiAnalyticalScheme/SemiAnalyticalSFINode.h"
#include <SemiAnalyticalScheme/Module/SemiAnalyticalDensitySolver.h>

#include "ObjIO/ObjLoader.h"
#include <BasicShapes/CubeModel.h>
#include <Commands/Merge.h>
#include <BasicShapes/PlaneModel.h>
#include <GLRenderEngine.h>
#include <Samplers/ShapeSampler.h>
#include <SemiAnalyticalScheme/TriangularMeshBoundary.h>

#include <Mapping/TextureMeshToTriangleSet.h>

#include <Volume/BasicShapeToVolume.h>
#include <Multiphysics/VolumeBoundary.h>
#include <ABCExporter/ParticleWriterABC.h>
#include <TriangleMeshWriter.h>

#include <Auxiliary/DataSource.h>
#include <ParticleSystem/Module/SurfaceEnergyForce.h>
#include <ParticleSystem/Viscosity/ImplicitViscosity.h>


using namespace std;
using namespace dyno;

#define SAMPLING_DISTANCE 0.02f
#define VELOCITYRENDER
//#define LARGETIME
std::shared_ptr<SceneGraph> createScene()
{
	Vec3f gravity = Vec3f(0.0f, -9.8f, 0.0f);
	gravity *= 0.1;
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setGravity(gravity);
	scn->setLowerBound(Vec3f(-2.0f));
	scn->setUpperBound(Vec3f(3.0f));
	//Create a cube box mode

	//Create turbing
	//auto turbing = scn->addNode(std::make_shared<ArticulatedBody<DataType3f>>());
	//turbing->varLocation()->setValue(Vec3f(0.0f, -0.125f, 0.0f));

	//std::string filename = getAssetPath() + "obj/Turbing/WaterTurbine_Turbine.obj";
	//turbing->varFilePath()->setValue(FilePath(filename));

	////first gear
	//RigidBodyInfo info;
	//info.position = Vec3f(0.0f, 0.0f, 0.0f);
	//info.angularVelocity = Vec3f(0, -1, 0);
	//info.motionType = BodyType::Kinematic;
	//info.bodyId = 0;
	//auto actor = turbing->createRigidBody(info);

	//CapsuleInfo capsule;
	//capsule.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
	//capsule.radius = 0.05f;
	//capsule.halfLength = 0.26f;

	//auto multibody = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
	//turbing->connect(multibody->importVehicles());

	//turbing->bindShape(actor, Pair<uint, uint>(0, 0));


	auto emitter = scn->addNode(std::make_shared<CircularEmitter<DataType3f>>());
	emitter->varLocation()->setValue(Vec3f(-1.85f, -0.016f, -1.434f));
	emitter->varRotation()->setValue(Vec3f(0.0f, 0.0f, 90.0f));
	emitter->varSamplingDistance()->setValue(0.02f);
	emitter->varVelocityMagnitude()->setValue(5.0f);
	emitter->varRadius()->setValue(0.45f);

#ifdef LARGETIME
	Real dt = 0.005;
#else
	Real dt = 0.001;
#endif // LARGETIME
	auto sfi = scn->addNode(std::make_shared<SemiAnalyticalParticleFluid<DataType3f>>());
	sfi->varSamplingDistance()->setValue(SAMPLING_DISTANCE);
	sfi->setDt(dt);
	sfi->varSmoothingLength()->setValue(1.2);
	sfi->varSearchRadius()->setValue(sfi->varSmoothingLength()->getValue() * sfi->varSamplingDistance()->getValue());
	{
		auto solver = sfi->animationPipeline()->findFirstModule<SemiAnalyticalDensitySolver<DataType3f>>();
		solver->varMu()->setValue(1);
		solver->varKappaLower()->setValue(100.0);
		solver->varPolynomialNumber()->setValue(3);
		solver->varIterationNumber()->setValue(5);
		solver->varD_hat()->setValue(SAMPLING_DISTANCE);
		auto surfacesolver = sfi->animationPipeline()->findFirstModule<SurfaceEnergyForce<DataType3f>>();
		surfacesolver->varKappa()->setValue(0.0);
		auto viscositysolver = sfi->animationPipeline()->findFirstModule<ImplicitViscosity<DataType3f>>();
		viscositysolver->varViscosity()->setValue(5.00);
	}
	emitter->connect(sfi->importParticleEmitters());

	{
		sfi->graphicsPipeline()->clear();
		auto ptRender = std::make_shared<GLPointVisualModule>();
		ptRender->varPointSize()->setValue(0.004);
		ptRender->varBaseColor()->setValue(Color(1, 0, 0));
		ptRender->varMetallic()->setValue(1.0f);
		ptRender->varRoughness()->setValue(1.0f);
		ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
		auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
		auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
#ifdef VELOCITYRENDER
#ifdef LARGETIME
		colorMapper->varMax()->setValue(0.7f);
#else
		colorMapper->varMax()->setValue(5.0f);
#endif // LARGETIME
		sfi->stateVelocity()->connect(calculateNorm->inVec());
		calculateNorm->outNorm()->connect(colorMapper->inScalar());
		sfi->graphicsPipeline()->pushModule(calculateNorm);
#else
		colorMapper->varMax()->setValue(1100.0f);
		colorMapper->varMin()->setValue(1000.0f);
		sfi->stateDensity()->connect(colorMapper->inScalar());
#endif // VELOCITYRENDER
		colorMapper->outColor()->connect(ptRender->inColor());
		sfi->statePointSet()->connect(ptRender->inPointSet());
		sfi->graphicsPipeline()->pushModule(colorMapper);
		sfi->graphicsPipeline()->pushModule(ptRender);
	}

	//tyre mode
	auto pipe = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
	pipe->varFileName()->setValue(getAssetPath() + "obj/Turbing/WaterTurbine_Pipeline2.obj");
	auto pipeVisualizer = pipe->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	pipeVisualizer->varAlpha()->setValue(0.2f);
	pipeVisualizer->varBaseColor()->setValue(Color::Aquamarine());

	auto turbing = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
	turbing->varFileName()->setValue(getAssetPath() + "obj/Turbing/WaterTurbine_Turbine.obj");
	//turbing->varLocation()->setValue(Vec3f(0.0f, -0.125f, 0.0f));
	turbing->varAngularVelocity()->setValue(Vec3f(0, -3, 0));
	turbing->setForceUpdate(true);
	auto turbingVisualizer = turbing->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	turbingVisualizer->varAlpha()->setValue(0.2f);
	turbingVisualizer->varBaseColor()->setValue(Color::Cornsilk());

	auto fan = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
	fan->varFileName()->setValue(getAssetPath() + "obj/Turbing/WaterTurbine_GuideVane.obj");
	auto fasVisualizer = fan->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	fasVisualizer->varAlpha()->setValue(0.2f);
	fasVisualizer->varBaseColor()->setValue(Color::Brown());

	auto fan_B = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
	fan_B->varFileName()->setValue(getAssetPath() + "obj/Turbing/WaterTurbine_GuideVane_B.obj");
	auto fasVisualizer_B = fan->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	fasVisualizer_B->varAlpha()->setValue(0.2f);
	fasVisualizer_B->varBaseColor()->setValue(Color::Brown());

	auto merge = scn->addNode(std::make_shared<Merge<DataType3f>>());
	merge->varUpdateMode()->setCurrentKey(1);
	merge->setVisible(false);

	pipe->outTriangleSet()->connect(merge->inTriangleSets());

	fan->outTriangleSet()->connect(merge->inTriangleSets());
	fan_B->outTriangleSet()->connect(merge->inTriangleSets());

	turbing->outTriangleSet()->connect(merge->inTriangleSets());


	//{
	//	auto transformer = std::make_shared<InstanceTransform<DataType3f>>();
	//	turbing->stateCenter()->connect(transformer->inCenter());
	//	turbing->stateBindingPair()->connect(transformer->inBindingPair());
	//	turbing->stateBindingTag()->connect(transformer->inBindingTag());
	//	turbing->stateRotationMatrix()->connect(transformer->inRotationMatrix());
	//	turbing->stateInstanceTransform()->connect(transformer->inInstanceTransform());
	//	turbing->animationPipeline()->pushModule(transformer);

	//	auto texMeshConverter = std::make_shared<TextureMeshToTriangleSet<DataType3f>>();
	//	turbing->stateTextureMesh()->connect(texMeshConverter->inTextureMesh());
	//	transformer->outInstanceTransform()->connect(texMeshConverter->inTransform());
	//	turbing->animationPipeline()->pushModule(texMeshConverter);

	//	turbing->animationPipeline()->promoteOutputToNode(texMeshConverter->outTriangleSet())->connect(merge->inTriangleSet03());
	//	turbing->animationPipeline()->promoteOutputToNode(texMeshConverter->outTriangleSet())->connect(sfi->inTriangleSet03());
	//}

	auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube->varLocation()->setValue(Vec3f(-2.611, 0.0f, -1.409));
	cube->varScale()->setValue(Vec3f(1.5f));
	cube->stateTriangleSet()->connect(merge->inTriangleSets());
	cube->graphicsPipeline()->clear();

	merge->stateTriangleSets()->connect(sfi->inTriangleSets());

	//auto meshWriter = std::make_shared <TriangleMeshWriter<DataType3f>>();
	//meshWriter->varOutputType()->getDataPtr()->setCurrentKey(0);
	//meshWriter->varPrefix()->setValue("A");
	//meshWriter->varOutputPath()->setValue((std::string)"D:\\ljyData2");
	//meshWriter->varStride()->setValue(8);
	//sfi->inTriangleSetMerge()->connect(meshWriter->inTopology());
	//sfi->stateFrameNumber()->connect(meshWriter->inFrameNumber());
	//sfi->animationPipeline()->pushModule(meshWriter);

	//auto meshWriter2 = std::make_shared <TriangleMeshWriter<DataType3f>>();
	//meshWriter2->varOutputPath()->setValue((std::string)"D:\\ljyData2");
	//meshWriter2->varOutputType()->getDataPtr()->setCurrentKey(0);
	//meshWriter2->varPrefix()->setValue("B");
	//meshWriter2->varStride()->setValue(8);
	//sfi->inTriangleSet03()->connect(meshWriter2->inTopology());
	//sfi->stateFrameNumber()->connect(meshWriter2->inFrameNumber());
	//sfi->animationPipeline()->pushModule(meshWriter2);

	//auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
	//sfi->stateVelocity()->connect(calculateNorm->inVec());
	//sfi->animationPipeline()->pushModule(calculateNorm);

	//auto ptcWriter = std::make_shared <ParticleWriterABC<DataType3f>>();
	//calculateNorm->outNorm()->connect(ptcWriter->inColor());
	//ptcWriter->varOutputPath()->setValue((std::string)"D:\\ljyData2\\C_");
	//sfi->statePosition()->connect(ptcWriter->inPosition());
	//sfi->stateFrameNumber()->connect(ptcWriter->inFrameNumber());
	//ptcWriter->varInterval()->setValue(8);
	//sfi->animationPipeline()->pushModule(ptcWriter);

	scn->printNodeInfo(true);
	scn->printSimulationInfo(true);
	return scn;
}

void RecieveLogMessage(const Log::Message& m)
{
	// ����log���
	switch (m.type)
	{
	case Log::Info:
		std::cout << ">>>: " << m.text << std::endl; break;
		// case Log::Warning:
		// 	cout << "???: " << m.text << endl; break;
		// case Log::Error:
		// 	cout << "!!!: " << m.text << endl; break;
	case Log::User:
		std::cout << ">>>: " << m.text << std::endl; break;
	default: break;
	}
}
int main()
{
	Modeling::initStaticPlugin();
	PaticleSystem::initStaticPlugin();
	SemiAnalyticalScheme::initStaticPlugin();

	QtApp app;
	app.setSceneGraph(createScene());
	app.initialize(1024, 768);

	app.renderWindow()->getCamera()->setEyePos(Vec3f(-2.74f, 3.64f, 1.15f));
	app.renderWindow()->getCamera()->setTargetPos(Vec3f(-0.19f, 0.14f, 0.03f));
	app.renderWindow()->setMainLightDirection(glm::vec3(-0.16f, -0.04f, 0.06f));
	auto renderer = std::dynamic_pointer_cast<dyno::GLRenderEngine>(app.renderWindow()->getRenderEngine());
	if (renderer) {
		renderer->setUseEnvmapBackground(false);

		renderer->bgColor0 = { 1, 1, 1 };
		renderer->bgColor1 = { 1, 1, 1 };

		renderer->planeColor = { 1,1,1,1 };
		renderer->rulerColor = { 1,1,1,1 };
		renderer->showGround = false;
		auto& light = app.renderWindow()->getRenderParams().light;
		light.mainLightScale = 10;
	}

	//Log::setUserReceiver(&RecieveLogMessage);
	app.mainLoop();

	return 0;
}
