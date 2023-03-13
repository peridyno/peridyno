#include <QtApp.h>

#include <SceneGraph.h>

#include <HeightField/Ocean.h>
#include <HeightField/OceanPatch.h>
#include <HeightField/CapillaryWave.h>
#include <HeightField/Coupling.h>
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
	auto mapper = std::make_shared<HeightFieldToTriangleSet<DataType3f>>();
	root->stateHeightField()->connect(mapper->inHeightField());
	root->graphicsPipeline()->pushModule(mapper);

// 	mapper->varScale()->setValue(0.01);
// 	mapper->varTranslation()->setValue(Vec3f(0, 0.2, 0));

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Vec3f(0, 0.2, 1.0));
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	root->graphicsPipeline()->pushModule(sRender);

	return scn;
}

int main()
{
	HeightFieldLibrary::initStaticPlugin();

	QtApp app;
	app.initialize(1024, 768);

	app.setSceneGraph(createScene());
	app.renderWindow()->getCamera()->setUnitScale(52);

	app.mainLoop();

	return 0;
}