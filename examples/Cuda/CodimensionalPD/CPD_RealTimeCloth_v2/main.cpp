#include <GlfwApp.h>
#include "Peridynamics/Cloth.h"
#include <SceneGraph.h>

#include <Multiphysics/VolumeBoundary.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include "TriangleMeshWriter.h"

#include <Volume/VolumeLoader.h>

#include "Peridynamics/CodimensionalPD.h"
#include "Peridynamics/Module/DragSurfaceInteraction.h"
#include "Peridynamics/Module/DragVertexInteraction.h"

#include "StaticMeshLoader.h"
using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setLowerBound(Vec3f(-1.5, -0.1, -1.5));
	scn->setUpperBound(Vec3f(1.5, 3, 1.5));
	auto object = scn->addNode(std::make_shared<StaticMeshLoader<DataType3f>>());
	object->varFileName()->setValue(getAssetPath() + "cloth_shell/v2/woman_model_smaller.obj");
	
	auto volLoader = scn->addNode(std::make_shared<VolumeLoader<DataType3f>>());
	volLoader->varFileName()->setValue(getAssetPath() + "cloth_shell/v2/woman_v2.sdf");

	auto boundary = scn->addNode(std::make_shared<VolumeBoundary<DataType3f>>());
	volLoader->connect(boundary->importVolumes());

	auto cloth = scn->addNode(std::make_shared<CodimensionalPD<DataType3f>>());
	cloth->loadSurface(getAssetPath() + "cloth_shell/v2/cloth_v2.obj");
	cloth->connect(boundary->importTriangularSystems()); 
	{
		auto solver = cloth->animationPipeline()->findFirstModule<CoSemiImplicitHyperelasticitySolver<DataType3f>>();

		solver->setContactMaxIte(20);
		solver->varIterationNumber()->setValue(10);

		solver->setS(0.1);
		solver->setXi(0.15);
		solver->setE(1000);
		solver->setK_bend(0.1);
	}

	{
		auto interaction = std::make_shared<DragVertexInteraction<DataType3f>>();
		interaction->varCacheEvent()->setValue(false);
		cloth->stateTriangleSet()->connect(interaction->inInitialTriangleSet());
		cloth->statePosition()->connect(interaction->inPosition());
		cloth->stateVelocity()->connect(interaction->inVelocity());
		cloth->stateAttribute()->connect(interaction->inAttribute());
		cloth->stateTimeStep()->connect(interaction->inTimeStep());
		cloth->animationPipeline()->pushModule(interaction);
	}

	auto surfaceRendererCloth = std::make_shared<GLSurfaceVisualModule>();
	surfaceRendererCloth->setColor(Color(0.08,0.021,0.0));
	surfaceRendererCloth->varUseVertexNormal()->setValue(true);
	auto surfaceRenderer = std::make_shared<GLSurfaceVisualModule>();
	
	surfaceRenderer->setColor(Color(1,1,0.6));
	surfaceRenderer->varUseVertexNormal()->setValue(true);
	cloth->stateTriangleSet()->connect(surfaceRendererCloth->inTriangleSet());
	object->stateTriangleSet()->connect(surfaceRenderer->inTriangleSet());
	cloth->graphicsPipeline()->pushModule(surfaceRendererCloth);
	object->graphicsPipeline()->pushModule(surfaceRenderer);
	cloth->setVisible(true);
	object->setVisible(true);

	scn->printNodeInfo(true);
	scn->printSimulationInfo(true);

	return scn;
}

void RecieveLogMessage(const Log::Message& m)
{
	switch (m.type)
	{
	case Log::Info:
		cout << ">>>: " << m.text << endl; break;
	case Log::Warning:
		cout << "???: " << m.text << endl; break;
	case Log::Error:
		cout << "!!!: " << m.text << endl; break;
	case Log::User:
		cout << ">>>: " << m.text << endl; break;
	default: break;
	}
}


int main()
{
	Log::setUserReceiver(&RecieveLogMessage);

	GlfwApp window;
	window.setSceneGraph(createScene());

	window.initialize(1024, 768);

	window.renderWindow()->setSelectionMode(RenderWindow::PRIMITIVE_MODE);

	window.mainLoop();

	return 0;
}