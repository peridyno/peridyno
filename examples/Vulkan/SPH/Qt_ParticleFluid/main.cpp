#include <QtApp.h>

#include <SceneGraph.h>

#include "ParticleSystem/initializeParticleSystem.h"
#include "ParticleSystem/SquareEmitter.h"

#include "ParticleSystem/ParticleFluid.h"

#include "GLPointVisualModule.h"
#include "GLWireframeVisualModule.h"

#include "Module/CalculateNorm.h"
#include "ColorMapping.h"

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto emitter = scn->addNode(std::make_shared<SquareEmitter>());
	emitter->varLocation()->setValue(Vec3f(0.0f, 0.5f, 0.0f));

	auto wireRender = std::make_shared<GLWireframeVisualModule>();
	wireRender->setColor(Color(0, 1, 0));
	emitter->stateOutline()->connect(wireRender->inEdgeSet());
	emitter->graphicsPipeline()->pushModule(wireRender);

	auto fluid = scn->addNode(std::make_shared<ParticleFluid>());
	emitter->connect(fluid->importParticleEmitters());

	auto calculateNorm = std::make_shared<CalculateNorm>();
	fluid->stateVelocity()->connect(calculateNorm->inVec());
	fluid->graphicsPipeline()->pushModule(calculateNorm);

	auto colorMapper = std::make_shared<ColorMapping>();
	colorMapper->varMax()->setValue(5.0f);
	calculateNorm->outNorm()->connect(colorMapper->inScalar());
	fluid->graphicsPipeline()->pushModule(colorMapper);

	auto pointRender = std::make_shared<GLPointVisualModule>();
	pointRender->varColorMode()->setCurrentKey(GLPointVisualModule::PER_VERTEX_SHADER);
	fluid->statePointSet()->connect(pointRender->inPointSet());
	colorMapper->outColor()->connect(pointRender->inColor());

	fluid->graphicsPipeline()->pushModule(pointRender);

	return scn;
}

int main()
{
	VkSystem::instance()->setAssetPath(getAssetPath());
	VkSystem::instance()->initialize(true);

	PaticleSystem::initStaticPlugin();

	{

		QtApp app;

		app.setSceneGraph(createScene());
		app.initialize(1024, 768);
		app.mainLoop();
	}
	VkSystem::instance()->destroy();

	return 0;
}