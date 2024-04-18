#include <QtApp.h>
#include "ObjIO/ObjLoader.h"
#include <QtApp.h>

#include <SceneGraph.h>

#include <ParticleSystem/ParticleFluid.h>
#include <ParticleSystem/StaticBoundary.h>
#include <ParticleSystem/SquareEmitter.h>

//#include <Multiphysics/SolidFluidCoupling.h>

#include <Module/CalculateNorm.h>
#include <Peridynamics/HyperelasticBody.h>
#include <Peridynamics/Cloth.h>
#include <Peridynamics/Thread.h>


#include "Node/GLPointVisualNode.h"
#include "Node/GLSurfaceVisualNode.h"

#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>


#include "RigidBody/RigidBody.h"
#include "ParticleSystem/StaticBoundary.h"
#include "ParticleSystem/SquareEmitter.h"
#include "ParticleSystem/CircularEmitter.h"
#include "ParticleSystem/ParticleFluid.h"

#include "Topology/TriangleSet.h"
#include "Collision/NeighborPointQuery.h"

#include "ParticleWriter.h"
#include "EigenValueWriter.h"

#include "Module/CalculateNorm.h"

#include <ColorMapping.h>


#include "SemiAnalyticalScheme/ComputeParticleAnisotropy.h"
#include "SemiAnalyticalScheme/SemiAnalyticalSFINode.h"
#include "SemiAnalyticalScheme/SemiAnalyticalPositionBasedFluidModel.h"
#include "SemiAnalyticalScheme/TriangularMeshBoundary.h"


#include "RigidBody/RigidBody.h"
#include "ParticleSystem/StaticBoundary.h"
#include "ParticleSystem/SquareEmitter.h"
#include "ParticleSystem/CircularEmitter.h"
#include "ParticleSystem/ParticleFluid.h"
#include "ParticleSystem/MakeParticleSystem.h"

#include "Topology/TriangleSet.h"
#include "Collision/NeighborPointQuery.h"

#include "ParticleWriter.h"
#include "EigenValueWriter.h"

#include "Module/CalculateNorm.h"

#include <ColorMapping.h>

#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLInstanceVisualModule.h>


#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>

#include <GLRenderEngine.h>

#include "SemiAnalyticalScheme/ComputeParticleAnisotropy.h"
#include "SemiAnalyticalScheme/SemiAnalyticalSFINode.h"
#include "SemiAnalyticalScheme/SemiAnalyticalPositionBasedFluidModel.h"

#include "StaticTriangularMesh.h"

#include "CubeModel.h"
#include "ParticleSystem/CubeSampler.h"
#include "SphereModel.h"
#include "Mapping/MergeTriangleSet.h"
#include "Mapping/TextureMeshToTriangleSet.h"
#include "Mapping/MergeTriangleSet.h"
#include "ColorMapping.h"

#include "ParticleSystem/Module/ParticleIntegrator.h"
#include "ParticleSystem/Module/ImplicitViscosity.h"
#include "ParticleSystem/Module/IterativeDensitySolver.h"

//Framework
#include "Auxiliary/DataSource.h"

#include "Collision/NeighborPointQuery.h"
#include "Collision/NeighborTriangleQuery.h"

#include "SemiAnalyticalScheme/SemiAnalyticalPBD.h"
#include "SemiAnalyticalScheme/TriangularMeshConstraint.h"

#include "Auxiliary/DataSource.h"

#include "GltfLoader.h"

#include "Node/GLSurfaceVisualNode.h"

using namespace dyno;

std::shared_ptr<SceneGraph> creatScene();
void importOtherModel(std::shared_ptr<SceneGraph> scn);

float total_scale = 6;

std::shared_ptr<SceneGraph> creatScene()
{	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();


	//***************************************Scene Setting***************************************//
	// Scene Setting
	scn->setTotalTime(3.0f);
	scn->setGravity(Vec3f(0.0f, -9.8f, 0.0f));
	scn->setLowerBound(Vec3f(-0.5f, 0.0f, -4.0f) * total_scale);
	scn->setUpperBound(Vec3f(0.5f, 1.0f, 4.0f) * total_scale);


	// Create Var
	Vec3f velocity = Vec3f(0,0,6);
	Color color = Color(1, 1, 1);

	Vec3f LocationBody = Vec3f(0, 0.01, -1);

	Vec3f anglurVel = Vec3f(100,0,0); 
	Vec3f scale = Vec3f(0.4,0.4,0.4);

	auto gltf = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	gltf->varFileName()->setValue(getAssetPath() + "Jeep/JeepGltf/jeep.gltf");


	// Import Road
	auto road = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	road->varFileName()->setValue(getAssetPath() + "Jeep/Road/Road.obj");
	road->varScale()->setValue(Vec3f(0.04) * total_scale);
	road->varLocation()->setValue(Vec3f(0, 0, 3.5));
	auto glRoad = road->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	glRoad->setColor(color);


	auto texMeshConverter = scn->addNode(std::make_shared<TextureMeshToTriangleSetNode<DataType3f>>());
	gltf->stateTextureMesh()->connect(texMeshConverter->inTextureMesh());

	auto tsMerger = scn->addNode(std::make_shared<MergeTriangleSet<DataType3f>>());
	//texMeshConverter->outTriangleSet()->connect(tsMerger->inFirst());
	road->outTriangleSet()->connect(tsMerger->inSecond());

	//*************************************** Cube Sample ***************************************//
	// Cube 
	auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube->varLocation()->setValue(Vec3f(0.0, 0.025, 0.4) * total_scale);
	cube->varLength()->setValue(Vec3f(0.35, 0.02, 3) * total_scale);
	cube->varScale()->setValue(Vec3f(2, 1, 1));
	cube->graphicsPipeline()->disable();

	auto cubeSmapler = scn->addNode(std::make_shared<CubeSampler<DataType3f>>());
	cubeSmapler->varSamplingDistance()->setValue(0.005f * total_scale);
	cube->outCube()->connect(cubeSmapler->inCube());
	cubeSmapler->graphicsPipeline()->disable();

	//MakeParticleSystem
	auto particleSystem = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());
	cubeSmapler->statePointSet()->promoteOuput()->connect(particleSystem->inPoints());

	//*************************************** Fluid ***************************************//
	//Particle fluid node
	auto fluid = scn->addNode(std::make_shared<ParticleFluid<DataType3f>>());

	{
		fluid->animationPipeline()->clear();

		auto smoothingLength = fluid->animationPipeline()->createModule<FloatingNumber<DataType3f>>();
		smoothingLength->varValue()->setValue(0.006f * total_scale);

		auto samplingDistance = fluid->animationPipeline()->createModule<FloatingNumber<DataType3f>>();
		samplingDistance->varValue()->setValue(Real(0.005) * total_scale);

		auto integrator = std::make_shared<ParticleIntegrator<DataType3f>>();
		fluid->stateTimeStep()->connect(integrator->inTimeStep());
		fluid->statePosition()->connect(integrator->inPosition());
		fluid->stateVelocity()->connect(integrator->inVelocity());
		fluid->stateForce()->connect(integrator->inForceDensity());
		fluid->animationPipeline()->pushModule(integrator);

		auto nbrQuery = std::make_shared<NeighborPointQuery<DataType3f>>();
		smoothingLength->outFloating()->connect(nbrQuery->inRadius());
		fluid->statePosition()->connect(nbrQuery->inPosition());
		fluid->animationPipeline()->pushModule(nbrQuery);

		auto density = std::make_shared<IterativeDensitySolver<DataType3f>>();
		density->varKappa()->setValue(0.1f);

		fluid->stateTimeStep()->connect(density->inTimeStep());
		fluid->statePosition()->connect(density->inPosition());
		fluid->stateVelocity()->connect(density->inVelocity());
		nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
		fluid->animationPipeline()->pushModule(density);

		smoothingLength->outFloating()->connect(density->inSmoothingLength());
		samplingDistance->outFloating()->connect(density->inSamplingDistance());

		auto viscosity = std::make_shared<ImplicitViscosity<DataType3f>>();
		viscosity->varViscosity()->setValue(Real(10.0));
		fluid->stateTimeStep()->connect(viscosity->inTimeStep());
		smoothingLength->outFloating()->connect(viscosity->inSmoothingLength());
		fluid->statePosition()->connect(viscosity->inPosition());
		fluid->stateVelocity()->connect(viscosity->inVelocity());
		nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
		fluid->animationPipeline()->pushModule(viscosity);
	}

	//Setup the point render
	{
		auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
		fluid->stateVelocity()->connect(calculateNorm->inVec());
		fluid->graphicsPipeline()->pushModule(calculateNorm);

		auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
		colorMapper->varMax()->setValue(5.0f);
		calculateNorm->outNorm()->connect(colorMapper->inScalar());
		fluid->graphicsPipeline()->pushModule(colorMapper);

		auto ptRender = std::make_shared<GLPointVisualModule>();
		ptRender->setColor(Color(1, 0, 0));
		ptRender->varPointSize()->setValue(0.01);
		ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);

		fluid->statePointSet()->connect(ptRender->inPointSet());
		colorMapper->outColor()->connect(ptRender->inColor());

		fluid->graphicsPipeline()->pushModule(ptRender);
	}

	particleSystem->connect(fluid->importInitialStates());

	//SemiAnalyticalSFINode
	auto meshBoundary = scn->addNode(std::make_shared<TriangularMeshBoundary<DataType3f>>());
	meshBoundary->varThickness()->setValue(0.005f * total_scale);

	fluid->connect(meshBoundary->importParticleSystems());
	tsMerger->stateTriangleSet()->connect(meshBoundary->inTriangleSet());

	//Create a boundary
	auto staticBoundary = scn->addNode(std::make_shared<StaticBoundary<DataType3f>>()); ;
	staticBoundary->loadCube(Vec3f(-4.6, 0, -7.2), Vec3f(4.6, 2, 12), 0.1, true);
	fluid->connect(staticBoundary->importParticleSystems());

	auto sVisual = scn->addNode(std::make_shared<GLSurfaceVisualNode<DataType3f>>());
	texMeshConverter->outTriangleSet()->connect(sVisual->inTriangleSet());

	//*************************************** Import Other Models ***************************************//

	//Other Models
//	importOtherModel(scn);

	return scn;
}

int main()
{
	QtApp window;
	window.setSceneGraph(creatScene());
	window.initialize(1366, 768);

	//Set the distance unit for the camera, the fault unit is meter
	window.renderWindow()->getCamera()->setUnitScale(3.0f);

	window.mainLoop();

	return 0;
}