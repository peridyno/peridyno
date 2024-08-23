#include <GlfwApp.h>

#include <SceneGraph.h>

#include <HeightField/GranularMedia.h>

#include "Mapping/HeightFieldToTriangleSet.h"

#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>

#include <HeightField/SurfaceParticleTracking.h>
#include <HeightField/RigidSandCoupling.h>

#include "GltfLoader.h"
#include "RigidBody/Vechicle.h"

#include "Collision/NeighborElementQuery.h"
#include "Collision/CollistionDetectionBoundingBox.h"

#include "RigidBody/Module/PJSConstraintSolver.h"

#include "RigidBody/Module/ContactsUnion.h"

#include <Module/GLPhotorealisticInstanceRender.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();


	auto gltf = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	gltf->setVisible(false);
	gltf->varFileName()->setValue(getAssetPath() + "Jeep/JeepGltf/jeep.gltf");


	auto jeep = scn->addNode(std::make_shared<Jeep<DataType3f>>());
	jeep->varLocation()->setValue(Vec3f(0.0f, 0.0f, -10.0f));

	//Replace the animation pipeline for jeep
	{
		jeep->animationPipeline()->clear();

		auto defaultTopo = std::make_shared<DiscreteElements<DataType3f>>();
		jeep->stateTopology()->setDataPtr(std::make_shared<DiscreteElements<DataType3f>>());

		auto elementQuery = std::make_shared<NeighborElementQuery<DataType3f>>();
		jeep->stateTopology()->connect(elementQuery->inDiscreteElements());
		jeep->stateCollisionMask()->connect(elementQuery->inCollisionMask());
		jeep->stateAttribute()->connect(elementQuery->inAttribute());
		jeep->animationPipeline()->pushModule(elementQuery);
		//elementQuery->varSelfCollision()->setValue(false);

		auto cdBV = std::make_shared<CollistionDetectionBoundingBox<DataType3f>>();
		jeep->stateTopology()->connect(cdBV->inDiscreteElements());
		jeep->animationPipeline()->pushModule(cdBV);

		auto merge = std::make_shared<ContactsUnion<DataType3f>>();
		elementQuery->outContacts()->connect(merge->inContactsA());
		cdBV->outContacts()->connect(merge->inContactsB());

		jeep->animationPipeline()->pushModule(merge);

		auto iterSolver = std::make_shared<PJSConstraintSolver<DataType3f>>();
		jeep->stateTimeStep()->connect(iterSolver->inTimeStep());
		jeep->varFrictionEnabled()->connect(iterSolver->varFrictionEnabled());
		jeep->varGravityEnabled()->connect(iterSolver->varGravityEnabled());
		jeep->varGravityValue()->connect(iterSolver->varGravityValue());
		jeep->varFrictionCoefficient()->connect(iterSolver->varFrictionCoefficient());
		jeep->varSlop()->connect(iterSolver->varSlop());
		jeep->stateMass()->connect(iterSolver->inMass());

		jeep->stateCenter()->connect(iterSolver->inCenter());
		jeep->stateVelocity()->connect(iterSolver->inVelocity());
		jeep->stateAngularVelocity()->connect(iterSolver->inAngularVelocity());
		jeep->stateRotationMatrix()->connect(iterSolver->inRotationMatrix());
		jeep->stateInertia()->connect(iterSolver->inInertia());
		jeep->stateQuaternion()->connect(iterSolver->inQuaternion());
		jeep->stateInitialInertia()->connect(iterSolver->inInitialInertia());
		jeep->stateTopology()->connect(iterSolver->inDiscreteElements());
		merge->outContacts()->connect(iterSolver->inContacts());
		jeep->animationPipeline()->pushModule(iterSolver);
	}

	gltf->stateTextureMesh()->connect(jeep->inTextureMesh());

	auto prRender = std::make_shared<GLPhotorealisticInstanceRender>();
	jeep->inTextureMesh()->connect(prRender->inTextureMesh());
	jeep->stateInstanceTransform()->connect(prRender->inTransform());
	jeep->graphicsPipeline()->pushModule(prRender);

	float spacing = 0.1f;
	uint res = 128;
	auto sand = scn->addNode(std::make_shared<GranularMedia<DataType3f>>());
	sand->varOrigin()->setValue(-0.5f * Vec3f(res * spacing, 0.0f, res * spacing));
	sand->varSpacing()->setValue(spacing);
	sand->varWidth()->setValue(res);
	sand->varHeight()->setValue(res);
	sand->varDepth()->setValue(0.2);
	sand->varDepthOfDiluteLayer()->setValue(0.1);

	auto mapper = std::make_shared<HeightFieldToTriangleSet<DataType3f>>();
	sand->stateHeightField()->connect(mapper->inHeightField());
	sand->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Color(0.8, 0.8, 0.8));
	sRender->varUseVertexNormal()->setValue(true);
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	sand->graphicsPipeline()->pushModule(sRender);

// 	auto tracking = scn->addNode(std::make_shared<SurfaceParticleTracking<DataType3f>>());
// 
// 	auto ptRender = tracking->graphicsPipeline()->findFirstModule<GLPointVisualModule>();
// 	ptRender->varPointSize()->setValue(0.01);
// 
// 	sand->connect(tracking->importGranularMedia());

	auto coupling = scn->addNode(std::make_shared<RigidSandCoupling<DataType3f>>());
	jeep->connect(coupling->importRigidBodySystem());
	sand->connect(coupling->importGranularMedia());

	return scn;
}

int main()
{
	GlfwApp app;
	app.initialize(1024, 768);

	app.setSceneGraph(createScene());
	app.renderWindow()->getCamera()->setUnitScale(5);

	app.mainLoop();

	return 0;
}