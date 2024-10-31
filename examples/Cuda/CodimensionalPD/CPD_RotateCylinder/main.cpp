#include <GlfwApp.h>
#include "Peridynamics/Cloth.h"
#include <SceneGraph.h>

#include <Multiphysics/VolumeBoundary.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include "TriangleMeshWriter.h"
#include "Peridynamics/CodimensionalPD.h"
#include "StaticTriangularMesh.h"
#include "ManualControl.h"
using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setLowerBound(Vec3f(-1.5, -1, -1.5));
	scn->setUpperBound(Vec3f(1.5, 3, 1.5));
	scn->setGravity(Vec3f(0));
	
	auto cloth = scn->addNode(std::make_shared<CodimensionalPD<DataType3f>>());
	cloth->loadSurface(getAssetPath() + "cloth_shell/cylinder400.obj");
	cloth->setDt(0.0005);
	{
		auto solver = cloth->animationPipeline()->findFirstModule<CoSemiImplicitHyperelasticitySolver<DataType3f>>();

		solver->setGrad_res_eps(0);
		solver->varIterationNumber()->setValue(10);

		solver->setS(0.1);
		solver->setXi(0.15);
		solver->setE(120);
		solver->setK_bend(0.001);
	}

	auto custom = std::make_shared<ManualControl<DataType3f>>();
	cloth->statePosition()->connect(custom->inPosition());
	cloth->stateVelocity()->connect(custom->inVelocity());
	cloth->stateFrameNumber()->connect(custom->inFrameNumber());
	cloth->stateAttribute()->connect(custom->inAttribute());
	cloth->animationPipeline()->pushModule(custom);

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
	GlfwApp window;
	window.setSceneGraph(createScene());

	window.initialize(1024, 768);
	window.mainLoop();

	return 0;
}