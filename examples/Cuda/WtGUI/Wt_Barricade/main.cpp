#include "WtApp.h"

#include "SceneGraph.h"

#include "ParticleSystem/Emitters/SquareEmitter.h"
#include "ParticleSystem/Emitters/CircularEmitter.h"
#include "ParticleSystem/ParticleFluid.h"

#include "Topology/TriangleSet.h"
#include "Mapping/MergeTriangleSet.h"

#include "Collision/NeighborPointQuery.h"

#include "Module/CalculateNorm.h"
#include "BasicShapes/CubeModel.h"

#include <ColorMapping.h>

#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLInstanceVisualModule.h>

#include "SemiAnalyticalScheme/ComputeParticleAnisotropy.h"
#include "SemiAnalyticalScheme/SemiAnalyticalSFINode.h"
#include "SemiAnalyticalScheme/TriangularMeshBoundary.h"
#include "SemiAnalyticalScheme/SemiAnalyticalPositionBasedFluidModel.h"

#include "StaticTriangularMesh.h"

#include "RigidBody/initializeRigidBody.h"
#include "ParticleSystem/initializeParticleSystem.h"
#include "DualParticleSystem/initializeDualParticleSystem.h"
#include "Peridynamics/initializePeridynamics.h"
#include "SemiAnalyticalScheme/initializeSemiAnalyticalScheme.h"
#include "Volume/initializeVolume.h"
#include "Multiphysics/initializeMultiphysics.h"
#include "HeightField/initializeHeightField.h"
#include "initializeModeling.h"
#include "initializeIO.h"

using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setTotalTime(3.0f);
	scn->setGravity(Vec3f(0.0f, -9.8f, 0.0f));
	scn->setLowerBound(Vec3f(-1.0f, 0.0f, 0.0f));
	scn->setUpperBound(Vec3f(1.0f, 1.0f, 1.0f));

	return scn;
}

int main(int argc, char** argv)
{
	Modeling::initStaticPlugin();
	RigidBody::initStaticPlugin();
	PaticleSystem::initStaticPlugin();
	HeightFieldLibrary::initStaticPlugin();
	DualParticleSystem::initStaticPlugin();
	Peridynamics::initStaticPlugin();
	SemiAnalyticalScheme::initStaticPlugin();
	Volume::initStaticPlugin();
	Multiphysics::initStaticPlugin();
	dynoIO::initStaticPlugin();

	WtApp app;

	app.setSceneGraphCreator(&createScene);
	app.setSceneGraph(createScene());

	app.mainLoop();

	return 0;
}