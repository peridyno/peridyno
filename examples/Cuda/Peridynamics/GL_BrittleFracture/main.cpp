#include <UbiApp.h>

#include <SceneGraph.h>

#include <BasicShapes/CubeModel.h>

#include <Volume/BasicShapeToVolume.h>

#include <Multiphysics/VolumeBoundary.h>

#include <Peridynamics/ElastoplasticBody.h>

#include <GLRenderEngine.h>
#include <GLSurfaceVisualModule.h>
#include <GLPointVisualModule.h>

#include <Node/GLSurfaceVisualNode.h>

#include <Mapping/PointSetToTriangleSet.h>

#include <SurfaceMeshLoader.h>

// Modeling
#include <BasicShapes/CubeModel.h>

// ParticleSystem
#include <Samplers/CubeSampler.h>

#include "Collision/NeighborPointQuery.h"

#include "ParticleSystem/MakeParticleSystem.h"
#include "ParticleSystem/Module/ParticleIntegrator.h"
#include "ParticleSystem/Module/ImplicitViscosity.h"

#include "Peridynamics/Module/FractureModule.h"

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	//Create a scene graph
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	//Create a cube
	auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube->setVisible(false);
	cube->varLocation()->setValue(Vec3f(0.2, 0.2, 0.0));
	cube->varLength()->setValue(Vec3f(0.1, 0.1, 0.1));
	cube->varSegments()->setValue(Vec3i(10, 10, 10));

	//Create a sampler
	auto sampler = scn->addNode(std::make_shared<CubeSampler<DataType3f>>());
	sampler->varSamplingDistance()->setValue(0.005);
	sampler->graphicsPipeline()->disable();

	cube->outCube()->connect(sampler->inCube());

	auto initialParticles = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());

	sampler->statePointSet()->promoteOuput()->connect(initialParticles->inPoints());

	//Create a elastoplastic object
	auto elastoplasticBody = scn->addNode(std::make_shared<ElastoplasticBody<DataType3f>>());	
	initialParticles->connect(elastoplasticBody->importSolidParticles());

	{
		elastoplasticBody->animationPipeline()->clear();

		auto integrator = std::make_shared<ParticleIntegrator<DataType3f>>();
		elastoplasticBody->stateTimeStep()->connect(integrator->inTimeStep());
		elastoplasticBody->statePosition()->connect(integrator->inPosition());
		elastoplasticBody->stateVelocity()->connect(integrator->inVelocity());
		elastoplasticBody->animationPipeline()->pushModule(integrator);

		auto nbrQuery = std::make_shared<NeighborPointQuery<DataType3f>>();
		elastoplasticBody->stateHorizon()->connect(nbrQuery->inRadius());
		elastoplasticBody->statePosition()->connect(nbrQuery->inPosition());
		elastoplasticBody->animationPipeline()->pushModule(nbrQuery);

		auto plasticity = std::make_shared<FractureModule<DataType3f>>();
		plasticity->varCohesion()->setValue(0.00001);
		elastoplasticBody->stateHorizon()->connect(plasticity->inHorizon());
		elastoplasticBody->stateTimeStep()->connect(plasticity->inTimeStep());
		elastoplasticBody->statePosition()->connect(plasticity->inY());
		elastoplasticBody->stateReferencePosition()->connect(plasticity->inX());
		elastoplasticBody->stateVelocity()->connect(plasticity->inVelocity());
		elastoplasticBody->stateBonds()->connect(plasticity->inBonds());
		nbrQuery->outNeighborIds()->connect(plasticity->inNeighborIds());
		elastoplasticBody->animationPipeline()->pushModule(plasticity);

		auto visModule = std::make_shared<ImplicitViscosity<DataType3f>>();
		visModule->varViscosity()->setValue(Real(1));
		elastoplasticBody->stateTimeStep()->connect(visModule->inTimeStep());
		elastoplasticBody->stateHorizon()->connect(visModule->inSmoothingLength());
		elastoplasticBody->statePosition()->connect(visModule->inPosition());
		elastoplasticBody->stateVelocity()->connect(visModule->inVelocity());
		nbrQuery->outNeighborIds()->connect(visModule->inNeighborIds());
		elastoplasticBody->animationPipeline()->pushModule(visModule);
	}

	auto elastoplasticBodyRenderer = std::make_shared<GLPointVisualModule>();
	elastoplasticBodyRenderer->varPointSize()->setValue(0.005);
	elastoplasticBodyRenderer->setColor(Color(1, 0.2, 1));
	elastoplasticBodyRenderer->setColorMapMode(GLPointVisualModule::PER_OBJECT_SHADER);
	elastoplasticBody->statePointSet()->connect(elastoplasticBodyRenderer->inPointSet());
	elastoplasticBody->stateVelocity()->connect(elastoplasticBodyRenderer->inColor());
	elastoplasticBody->graphicsPipeline()->pushModule(elastoplasticBodyRenderer);

	//Create a container
	auto cubeBoundary = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cubeBoundary->varLocation()->setValue(Vec3f(0.5f, 1.0f, 0.5f));
	cubeBoundary->varLength()->setValue(Vec3f(2.0f));
	cubeBoundary->setVisible(false);

	auto cube2vol = scn->addNode(std::make_shared<BasicShapeToVolume<DataType3f>>());
	cube2vol->varGridSpacing()->setValue(0.02f);
	cube2vol->varInerted()->setValue(true);
	cubeBoundary->connect(cube2vol->importShape());

	auto container = scn->addNode(std::make_shared<VolumeBoundary<DataType3f>>());
	container->varTangentialFriction()->setValue(0.95f);
	cube2vol->connect(container->importVolumes());

	elastoplasticBody->connect(container->importParticleSystems());
	//elasticBody->connect(container->importParticleSystems());

	return scn;
}

int main(int argc, char* argv[])
{
	UbiApp app(GUIType::GUI_QT);
	app.setSceneGraph(createScene());
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}


