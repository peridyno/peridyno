#include <QtApp.h>

#include <SceneGraph.h>

#include <HeightField/Ocean.h>
#include <HeightField/OceanPatch.h>
#include <HeightField/CapillaryWave.h>
#include <HeightField/RigidWaterCoupling.h>
#include <HeightField/initializeHeightField.h>
#include <RigidBody/RigidBodySystem.h>
#include "Mapping/HeightFieldToTriangleSet.h"

#include <GLRenderEngine.h>
#include <GLSurfaceVisualModule.h>


#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Mapping/ContactsToEdgeSet.h>
#include <Mapping/ContactsToPointSet.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto root = scn->addNode(std::make_shared<OceanPatch<DataType3f>>());

	return scn;
}

int main()
{
	HeightFieldLibrary::initStaticPlugin();

	QtApp app;
	app.setSceneGraph(createScene());
	app.initialize(1024, 768);
	app.renderWindow()->getCamera()->setUnitScale(52);

	app.mainLoop();

	return 0;
}
