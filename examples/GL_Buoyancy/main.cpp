#include <GlfwApp.h>

#include <SceneGraph.h>

#include <HeightField/Ocean.h>
#include <HeightField/OceanPatch.h>
#include <HeightField/CapillaryWave.h>
#include <HeightField/Coupling.h>
#include <RigidBody/RigidBodySystem.h>

#include "Mapping/HeightFieldToTriangleSet.h"
#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Mapping/ContactsToEdgeSet.h>
#include <Mapping/ContactsToPointSet.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

#include "Collision/NeighborElementQuery.h"
using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto ocean = scn->addNode(std::make_shared<Ocean<DataType3f>>());

	auto oceanPatch = scn->addNode(std::make_shared<OceanPatch<DataType3f>>(512, 512, 8));
	oceanPatch->connect(ocean->importOceanPatch());

	auto capillaryWave = scn->addNode(std::make_shared<CapillaryWave<DataType3f>>(512, 512.0f));
	capillaryWave->connect(ocean->importCapillaryWaves());


	auto mapper = std::make_shared<HeightFieldToTriangleSet<DataType3f>>();
	mapper->varScale()->setValue(0.01);
	mapper->varTranslation()->setValue(Vec3f(0, 0.2, 0));

	ocean->stateTopology()->connect(mapper->inHeightField());
	ocean->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Vec3f(0, 0.2, 1.0));
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	ocean->graphicsPipeline()->pushModule(sRender);


	//rigid------------------------------------------------------------------
	auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());
	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(0, 0, 0);
	BoxInfo box;


	box.center = 0.5f * Vec3f(0, 0.4, 0);
	box.halfLength = Vec3f(0.1, 0.1, 0.1);
	rigid->addBox(box, rigidBody);


	auto Rmapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(Rmapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(Rmapper);


	auto rRender = std::make_shared<GLWireframeVisualModule>();
	rRender->setColor(Vec3f(1, 1, 0));
	Rmapper->outTriangleSet()->connect(rRender->inEdgeSet());
	rigid->graphicsPipeline()->pushModule(rRender);
	
	//coupling---------------------------------------
	auto trail = scn->addNode(std::make_shared<CapillaryWave<DataType3f>>(512, 512.0f));



	auto coupling = scn->addNode(std::make_shared<Coupling<DataType3f>>());
	rigid->connect(coupling->importRigidBodySystem());
	ocean->connect(coupling->importOcean());

	Rmapper->outTriangleSet()->connect(coupling->inTriangleSet());
	coupling->initialize();



	return scn;
}

int main()
{
	GlfwApp window;
	window.setSceneGraph(createScene());
	window.createWindow(1024, 768);
	window.mainLoop();

	return 0;
}
