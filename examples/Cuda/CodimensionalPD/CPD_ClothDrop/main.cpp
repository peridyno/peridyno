#include <UbiApp.h>
#include "Peridynamics/Cloth.h"
#include <SceneGraph.h>
#include <Log.h>
#include <ParticleSystem/StaticBoundary.h>
#include <Multiphysics/VolumeBoundary.h>
#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>

#include "Peridynamics/CodimensionalPD.h"
#include "../Modeling/StaticTriangularMesh.h"
#include "ManualControl.h"

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setLowerBound(Vec3f(-1.5, -1, -1.5));
	scn->setUpperBound(Vec3f(1.5, 3, 1.5));
	scn->setGravity(Vec3f(0,-200,0));

	auto cloth = scn->addNode(std::make_shared<CodimensionalPD<DataType3f>>());
	//also try:
	//auto cloth = scn->addNode(std::make_shared<CodimensionalPD<DataType3f>>(0.15, 1200, 0.001, 0.0001));
	//auto cloth = scn->addNode(std::make_shared<CodimensionalPD<DataType3f>>(0.15, 12000, 0.001, 0.0001));
	cloth->loadSurface(getAssetPath() + "cloth_shell/mesh_drop.obj");

	auto custom = std::make_shared<ManualControl<DataType3f>>();
	cloth->statePosition()->connect(custom->inPosition());
	cloth->stateVelocity()->connect(custom->inVelocity());
	cloth->stateFrameNumber()->connect(custom->inFrameNumber());
	cloth->stateAttribute()->connect(custom->inAttribute());
	cloth->animationPipeline()->pushModule(custom);
	//cloth->setSelfContact(false);
	
	auto surfaceRendererCloth = std::make_shared<GLSurfaceVisualModule>();
	surfaceRendererCloth->setColor(Color(1, 1, 1));

	cloth->stateTriangleSet()->connect(surfaceRendererCloth->inTriangleSet());
	cloth->graphicsPipeline()->pushModule(surfaceRendererCloth);
	cloth->setVisible(true);

	scn->printNodeInfo(true);
	scn->printSimulationInfo(true);

	return scn;
}

int main()
{
	UbiApp window(GUIType::GUI_QT);
	window.setSceneGraph(createScene());

	window.initialize(1024, 768);
	window.mainLoop();

	return 0;
}