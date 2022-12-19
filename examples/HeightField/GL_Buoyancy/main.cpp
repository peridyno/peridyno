#include <GlfwApp.h>

#include <SceneGraph.h>

#include <HeightField/Ocean.h>
#include <HeightField/OceanPatch.h>
#include <HeightField/CapillaryWave.h>
#include <HeightField/Coupling.h>
#include <HeightField/Boat.h>

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

	auto oceanPatch = scn->addNode(std::make_shared<OceanPatch<DataType3f>>());
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
	

	auto boat = scn->addNode(std::make_shared<Boat<DataType3f>>());
	boat->varFileName()->setValue(getAssetPath() + "bunny/sparse_bunny_mesh.obj");

	
	auto coupling = scn->addNode(std::make_shared<Coupling<DataType3f>>());
	boat->connect(coupling->importBoat());
	ocean->connect(coupling->importOcean());
	
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
