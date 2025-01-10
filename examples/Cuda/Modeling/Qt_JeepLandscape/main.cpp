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
#include <RigidBody/MultibodySystem.h>
#include <RigidBody/Module/InstanceTransform.h>

#include <Mapping/TextureMeshToTriangleSet.h>
#include <Mapping/MergeTriangleSet.h>
#include "ObjIO/ObjLoader.h"

using namespace dyno;

std::shared_ptr<SceneGraph> creatScene();
void importOtherModel(std::shared_ptr<SceneGraph> scn);

float total_scale = 6;

std::shared_ptr<SceneGraph> creatScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto jeep = scn->addNode(std::make_shared<Jeep<DataType3f>>());

	auto multibody = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
	jeep->connect(multibody->importVehicles());
	jeep->varLocation()->setValue(Vec3f(0,1,-5));

	auto ObjLand = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
	ObjLand->varFileName()->setValue(getAssetPath() + "landscape/Landscape_resolution_1000_1000.obj");
	ObjLand->varScale()->setValue(Vec3f(6));
	ObjLand->varLocation()->setValue(Vec3f(0, 0, 0.5));
	auto glLand = ObjLand->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	glLand->varBaseColor()->setValue(Color::LightGray());
	glLand->varUseVertexNormal()->setValue(true);
	
	ObjLand->outTriangleSet()->connect(multibody->inTriangleSet());
	


	return scn;
}

int main()
{
	QtApp window;
	window.setSceneGraph(creatScene());
	window.initialize(1366, 768);

	//Set the distance unit for the camera, the fault unit is meter
	window.renderWindow()->getCamera()->setUnitScale(10.0f);

	auto renderer = std::dynamic_pointer_cast<dyno::GLRenderEngine>(window.renderWindow()->getRenderEngine());
	if (renderer) {
		renderer->setEnvStyle(EEnvStyle::Studio);
		renderer->showGround = false;
		renderer->setUseEnvmapBackground(false);

	}

	window.mainLoop();

	return 0;
}