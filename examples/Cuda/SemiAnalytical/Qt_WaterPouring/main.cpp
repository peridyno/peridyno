#include "UbiApp.h"
#include "SceneGraph.h"

#include "ParticleSystem/Emitters/SquareEmitter.h"
#include "ParticleSystem/Emitters/CircularEmitter.h"
#include "ParticleSystem/ParticleFluid.h"

#include "SemiAnalyticalScheme/SemiAnalyticalSFINode.h"

#include "Mapping/MergeTriangleSet.h"

#include "Collision/NeighborPointQuery.h"

#include "Module/CalculateNorm.h"

#include <ColorMapping.h>

#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLInstanceVisualModule.h>

#include "BasicShapes/PlaneModel.h"
#include "BasicShapes/SphereModel.h"

#include "initializeModeling.h"
#include "ParticleSystem/initializeParticleSystem.h"
#include "SemiAnalyticalScheme/initializeSemiAnalyticalScheme.h"
#include "ObjIO/ObjLoader.h"

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setGravity(Vec3f(0.0f, -9.8f, 0.0f));

	//Create a particle emitter
	auto emitter = scn->addNode(std::make_shared<CircularEmitter<DataType3f>>());
	emitter->varLocation()->setValue(Vec3f(0.0f, 1.0f, 0.0f));

	//Setup boundaries
	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->varScale()->setValue(Vec3f(2.0f, 0.0f, 2.0f));

	auto objLoader = scn->addNode(std::make_shared<ObjLoader<DataType3f>>());
	objLoader->varFileName()->setValue(getAssetPath() + "Obj/Cup.obj");

	objLoader->varLocation()->setValue(Vec3f(0, 0, 0));
	objLoader->varScale()->setValue(Vec3f(0.2, 0.2, 0.2));
	//objLoader->varCenter()->setValue(Vec3f(0.0f, 0.255f, 0.630f));
	objLoader->varAngularVelocity()->setValue(Vec3f(0, 2.0f, 0.0f));
	//objLoader->varVelocity()->setValue(Vec3f(0.0f, 0.0f, 2.0f));

	objLoader->graphicsPipeline()->clear();
	objLoader->setForceUpdate(true);


	auto merge = scn->addNode(std::make_shared<MergeTriangleSet<DataType3f>>());
	plane->stateTriangleSet()->connect(merge->inFirst());
	objLoader->outTriangleSet()->connect(merge->inSecond());

	auto sRenderf = std::make_shared<GLSurfaceVisualModule>();
	sRenderf->setColor(Color(1.0f, 1.0f, 1.0f));
	sRenderf->setAlpha(0.2f);
	sRenderf->setVisible(true);
	merge->stateTriangleSet()->connect(sRenderf->inTriangleSet());
	merge->graphicsPipeline()->pushModule(sRenderf);


	//SFI node
	auto sfi = scn->addNode(std::make_shared<SemiAnalyticalSFINode<DataType3f>>());

	emitter->connect(sfi->importParticleEmitters());
	merge->stateTriangleSet()->connect(sfi->inTriangleSet());

	return scn;
}

int main()
{
	Modeling::initStaticPlugin();
	PaticleSystem::initStaticPlugin();
	SemiAnalyticalScheme::initStaticPlugin();

	UbiApp app(GUIType::GUI_QT);
	app.setSceneGraph(createScene());
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}