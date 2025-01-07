#include <UbiApp.h>

#include <SceneGraph.h>

#include <HeightField/GranularMedia.h>

#include "Mapping/HeightFieldToTriangleSet.h"

#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>

#include <HeightField/SurfaceParticleTracking.h>
#include <HeightField/RigidSandCoupling.h>

#include "GltfLoader.h"

#include "Collision/NeighborElementQuery.h"
#include "Collision/CollistionDetectionBoundingBox.h"

#include "RigidBody/Vehicle.h"
#include <RigidBody/MultibodySystem.h>
#include "RigidBody/Module/PJSConstraintSolver.h"
#include "RigidBody/Module/ContactsUnion.h"

#include <Module/GLPhotorealisticInstanceRender.h>
#include "BasicShapes/PlaneModel.h"
#include <GLRenderEngine.h>


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


using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	//build multibodySystem
	auto jeep = scn->addNode(std::make_shared<Jeep<DataType3f>>());
	jeep->varLocation()->setValue(Vec3f(0.0f, 0.0f, -10.0f));

	auto tank = scn->addNode(std::make_shared<Tank<DataType3f>>());
	tank->varLocation()->setValue(Vec3f(-6.0f, 0.0f, -10.0f));

	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->varLengthX()->setValue(40);
	plane->varLengthZ()->setValue(40);
	plane->varSegmentX()->setValue(20);
	plane->varSegmentZ()->setValue(20);

	auto multibody = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
	plane->stateTriangleSet()->connect(multibody->inTriangleSet());
	jeep->connect(multibody->importVehicles());
	tank->connect(multibody->importVehicles());

	float spacing = 0.1f;
	uint res = 512;
	auto sand = scn->addNode(std::make_shared<GranularMedia<DataType3f>>());
	sand->varOrigin()->setValue(-0.5f * Vec3f(res * spacing, 0.0f, res * spacing));
	sand->varSpacing()->setValue(spacing);
	sand->varWidth()->setValue(res);
	sand->varHeight()->setValue(res);
	sand->varDepth()->setValue(0.2);
	sand->varDepthOfDiluteLayer()->setValue(0.1);


	auto coupling = scn->addNode(std::make_shared<RigidSandCoupling<DataType3f>>());
	multibody->connect(coupling->importRigidBodySystem());
	sand->connect(coupling->importGranularMedia());

	return scn;
}

int main()
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

	UbiApp app(GUIType::GUI_QT);

	app.setSceneGraph(createScene());

	app.initialize(1024, 768);

	app.renderWindow()->getCamera()->setUnitScale(5);
	app.renderWindow()->getCamera()->setEyePos(Vec3f(1.67, 0.73, 0.7));
	app.renderWindow()->getCamera()->setTargetPos(Vec3f(-0.03, 0.14, -0.06));
	// setup envmap
	auto renderer = std::dynamic_pointer_cast<dyno::GLRenderEngine>(app.renderWindow()->getRenderEngine());
	if (renderer) {
		renderer->setEnvStyle(EEnvStyle::Studio);
		renderer->setEnvmapScale(0.24);
		renderer->setUseEnvmapBackground(false);
	}
	app.mainLoop();

	return 0;
}