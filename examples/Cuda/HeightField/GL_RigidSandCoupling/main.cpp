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

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();


// 	auto gltf = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
// 	gltf->setVisible(false);
// 	gltf->varFileName()->setValue(getAssetPath() + "Jeep/JeepGltf/jeep.gltf");


	auto jeep = scn->addNode(std::make_shared<Jeep<DataType3f>>());
	jeep->varLocation()->setValue(Vec3f(0.0f, 0.0f, -10.0f));

	auto multibody = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());

	jeep->connect(multibody->importVehicles());

	//Replace the animation pipeline for jeep
	{
		multibody->animationPipeline()->clear();

		auto defaultTopo = std::make_shared<DiscreteElements<DataType3f>>();
		multibody->stateTopology()->setDataPtr(std::make_shared<DiscreteElements<DataType3f>>());

		auto elementQuery = std::make_shared<NeighborElementQuery<DataType3f>>();
		multibody->stateTopology()->connect(elementQuery->inDiscreteElements());
		multibody->stateCollisionMask()->connect(elementQuery->inCollisionMask());
		multibody->stateAttribute()->connect(elementQuery->inAttribute());
		multibody->animationPipeline()->pushModule(elementQuery);
		//elementQuery->varSelfCollision()->setValue(false);

		auto cdBV = std::make_shared<CollistionDetectionBoundingBox<DataType3f>>();
		multibody->stateTopology()->connect(cdBV->inDiscreteElements());
		multibody->animationPipeline()->pushModule(cdBV);

		auto merge = std::make_shared<ContactsUnion<DataType3f>>();
		elementQuery->outContacts()->connect(merge->inContactsA());
		cdBV->outContacts()->connect(merge->inContactsB());

		multibody->animationPipeline()->pushModule(merge);

		auto iterSolver = std::make_shared<PJSConstraintSolver<DataType3f>>();
		multibody->stateTimeStep()->connect(iterSolver->inTimeStep());
		multibody->varFrictionEnabled()->connect(iterSolver->varFrictionEnabled());
		multibody->varGravityEnabled()->connect(iterSolver->varGravityEnabled());
		multibody->varGravityValue()->connect(iterSolver->varGravityValue());
		multibody->varFrictionCoefficient()->connect(iterSolver->varFrictionCoefficient());
		multibody->varSlop()->connect(iterSolver->varSlop());
		multibody->stateMass()->connect(iterSolver->inMass());

		multibody->stateCenter()->connect(iterSolver->inCenter());
		multibody->stateVelocity()->connect(iterSolver->inVelocity());
		multibody->stateAngularVelocity()->connect(iterSolver->inAngularVelocity());
		multibody->stateRotationMatrix()->connect(iterSolver->inRotationMatrix());
		multibody->stateInertia()->connect(iterSolver->inInertia());
		multibody->stateQuaternion()->connect(iterSolver->inQuaternion());
		multibody->stateInitialInertia()->connect(iterSolver->inInitialInertia());
		multibody->stateTopology()->connect(iterSolver->inDiscreteElements());
		merge->outContacts()->connect(iterSolver->inContacts());
		multibody->animationPipeline()->pushModule(iterSolver);
	}

	//gltf->stateTextureMesh()->connect(jeep->inTextureMesh());

// 	auto prRender = std::make_shared<GLPhotorealisticInstanceRender>();
// 	jeep->inTextureMesh()->connect(prRender->inTextureMesh());
// 	jeep->stateInstanceTransform()->connect(prRender->inTransform());
// 	jeep->graphicsPipeline()->pushModule(prRender);

	float spacing = 0.1f;
	uint res = 128;
	auto sand = scn->addNode(std::make_shared<GranularMedia<DataType3f>>());
	sand->varOrigin()->setValue(-0.5f * Vec3f(res * spacing, 0.0f, res * spacing));
	sand->varSpacing()->setValue(spacing);
	sand->varWidth()->setValue(res);
	sand->varHeight()->setValue(res);
	sand->varDepth()->setValue(0.2);
	sand->varDepthOfDiluteLayer()->setValue(0.1);

// 	auto tracking = scn->addNode(std::make_shared<SurfaceParticleTracking<DataType3f>>());
// 
// 	auto ptRender = tracking->graphicsPipeline()->findFirstModule<GLPointVisualModule>();
// 	ptRender->varPointSize()->setValue(0.01);
// 
// 	sand->connect(tracking->importGranularMedia());

	auto coupling = scn->addNode(std::make_shared<RigidSandCoupling<DataType3f>>());
	multibody->connect(coupling->importRigidBodySystem());
	sand->connect(coupling->importGranularMedia());

	return scn;
}

int main()
{
	UbiApp app(GUIType::GUI_GLFW);

	app.setSceneGraph(createScene());

	app.initialize(1024, 768);

	app.renderWindow()->getCamera()->setUnitScale(5);

	app.mainLoop();

	return 0;
}