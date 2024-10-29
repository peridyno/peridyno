#include "QtGUI/QtApp.h"

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

	auto sphere = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	sphere->varLocation()->setValue(Vec3f(0.0f, 0.5f, 0.0f));
	sphere->varScale()->setValue(Vec3f(0.2f, 0.2f, 0.2f));

	auto merge = scn->addNode(std::make_shared<MergeTriangleSet<DataType3f>>());
	plane->stateTriangleSet()->connect(merge->inFirst());
	sphere->stateTriangleSet()->connect(merge->inSecond());

	//SFI node
	auto sfi = scn->addNode(std::make_shared<SemiAnalyticalSFINode<DataType3f>>());
	{
		auto ptRender = std::make_shared<GLPointVisualModule>();
		ptRender->varPointSize()->setValue(0.002);
		ptRender->setColor(Color(1, 0, 0));
		ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);

		auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
		auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
		colorMapper->varMax()->setValue(5.0f);
		sfi->stateVelocity()->connect(calculateNorm->inVec());
		calculateNorm->outNorm()->connect(colorMapper->inScalar());

		colorMapper->outColor()->connect(ptRender->inColor());
		sfi->statePointSet()->connect(ptRender->inPointSet());

		sfi->graphicsPipeline()->pushModule(calculateNorm);
		sfi->graphicsPipeline()->pushModule(colorMapper);
		sfi->graphicsPipeline()->pushModule(ptRender);
	}

	emitter->connect(sfi->importParticleEmitters());
	merge->stateTriangleSet()->connect(sfi->inTriangleSet());

	return scn;
}

int main()
{
	Modeling::initStaticPlugin();
	PaticleSystem::initStaticPlugin();
	SemiAnalyticalScheme::initStaticPlugin();

	QtApp app;
	app.setSceneGraph(createScene());
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}