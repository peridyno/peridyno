#include "WtApp.h"

#include "SceneGraph.h"

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

#include "ObjIO/initializeObjIO.h"
#include "ObjIO/ObjLoader.h"
#include "RigidBody/MultibodySystem.h"
#include <BasicShapes/CubeModel.h>
#include <BasicShapes/PlaneModel.h>

using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(1.5, 1, 1.5));
	scn->setLowerBound(Vec3f(-0.5, 0, -0.5));

	//auto cube1 = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());

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
	ObjIO::initStaticPlugin();

	WtApp app;

	app.setSceneGraphCreator(&createScene);
	app.setSceneGraph(createScene());
	app.mainLoop();

	return 0;
}