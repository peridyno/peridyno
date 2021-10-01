#include <GlfwApp.h>

#include <SceneGraph.h>

#include <RigidBody/RigidBodySystem.h>

#include <Module/CalculateNorm.h>
#include <Quat.h>

#include <GLRenderEngine.h>
#include <GLElementVisualModule.h>
#include <ColorMapping.h>


#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <GLSurfaceVisualModule.h>


using namespace std;
using namespace dyno;

void creat_scene_fluid()
{
	Log::sendMessage(Log::Info, "Simulation start");

	SceneGraph& scene = SceneGraph::getInstance();



	std::shared_ptr<RigidBodySystem<DataType3f>> rigid = scene.createNewScene<RigidBodySystem<DataType3f>>();

	std::shared_ptr<DiscreteElements<DataType3f>> DE = std::make_shared<DiscreteElements<DataType3f>>();
	


	for (int i = 8; i > 1; i--)
		for (int j = 0; j < i + 1; j++)
		{
			DE->addBox(Box3D(0.5f * Vec3f(0.5f, 1.1 - 0.13 * i, 0.12f + 0.2 * j + 0.1 * (8 - i)),
				Vec3f(1.0, 0.0, 0.0), Vec3f(0.0, 1.0, 0.0), Vec3f(0.0, 0.0, 1.0),
				0.5f * Vec3f(0.065, 0.065, 0.1)));

			rigid->host_angular_velocity.push_back(Vec3f(0));
			rigid->host_velocity.push_back(Vec3f(0.5, 0, 0));
			rigid->host_mass.push_back(0.004f);
			rigid->host_inertia_tensor.push_back(
				0.004 / 12.0f *
				Matrix3D(
					(0.065 * 0.065 + 0.1 * 0.1), 0, 0,
					0, (0.065 * 0.065 + 0.1 * 0.1), 0,
					0, 0, (0.065 * 0.065 + 0.065 * 0.065)
				)
			);
		}




	DE->initialize();

	printf("222\n");
	rigid->currentTopology()->setDataPtr(DE);

	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->currentTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Vec3f(1, 1, 0));
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	rigid->graphicsPipeline()->pushModule(sRender);
// 	auto eRender = std::make_shared<GLElementVisualModule>();
// 	eRender->discreteSet = DE;
// 	rigid->varTimeStep()->connect(eRender->inTimeStep());
// 	
// 	//rigid->addVisualModule(eRender);
// 	eRender->setColor(Vec3f(1, 0, 0));
// 
// 	rigid->graphicsPipeline()->pushModule(eRender);
	
	//rigid->initialize();
	

	GLRenderEngine* engine = new GLRenderEngine;

	GlfwApp window;
	window.setRenderEngine(engine);
	window.createWindow(1280, 768);
	window.mainLoop();

	delete engine;
}



int main()
{

	creat_scene_fluid();
	//test_render();

	return 0;
}


