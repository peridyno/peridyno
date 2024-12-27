#include <UbiApp.h>

#include <SceneGraph.h>

#include <RigidBody/ArticulatedBody.h>
#include <RigidBody/MultibodySystem.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Mapping/ContactsToEdgeSet.h>
#include <Mapping/ContactsToPointSet.h>
#include <Mapping/AnchorPointToPointSet.h>

#include "Collision/NeighborElementQuery.h"
#include "Collision/CollistionDetectionTriangleSet.h"
#include "Collision/CollistionDetectionBoundingBox.h"

#include <Module/GLPhotorealisticInstanceRender.h>

#include <BasicShapes/PlaneModel.h>
#include <map>

#include "GltfLoader.h"
//#define USE_HINGE

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> creatCar()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto jeep = scn->addNode(std::make_shared<ArticulatedBody<DataType3f>>());
	jeep->varFilePath()->setValue(getAssetPath() + "jeep_bridge/Jeep_Bridge.gltf");

	auto multibody = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
	jeep->connect(multibody->importVehicles());

// 	auto gltf = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
// 	gltf->setVisible(false);
// 	gltf->varFileName()->setValue(getAssetPath() + "jeep_bridge/Jeep_Bridge.gltf");
// 
// 	gltf->stateTextureMesh()->connect(jeep->inTextureMesh());
	auto texMesh = jeep->stateTextureMesh()->constDataPtr();
	std::vector<int> wheel_Id = { 0, 1, 2, 3, 4 };
	std::map<int, std::shared_ptr<PdActor>> Actors;
	RigidBodyInfo rigidbody;
	rigidbody.bodyId = 0;
	for (auto it : wheel_Id)
	{
		if (it != 4)
		{
			auto up = texMesh->shapes()[it]->boundingBox.v1;
			auto down = texMesh->shapes()[it]->boundingBox.v0;
			/*SphereInfo sphere;
			sphere.center = texMesh->shapes()[it]->boundingTransform.translation();
			sphere.radius = std::abs(up.y - down.y) / 2;
			Actors[it] = jeep->addSphere(sphere, rigidbody, 100);*/
			CapsuleInfo capsule;
			capsule.center = Vec3f(0.0f);
			capsule.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
			capsule.radius = std::abs(up.y - down.y) / 2;
			capsule.halfLength = 0.1495;
			rigidbody.position = texMesh->shapes()[it]->boundingTransform.translation();
			Actors[it] = jeep->addCapsule(capsule, rigidbody, 100);
			jeep->bind(Actors[it], Pair<uint, uint>(it, 0));
		}
		else
		{
			auto up = texMesh->shapes()[it]->boundingBox.v1;
			auto down = texMesh->shapes()[it]->boundingBox.v0;
			BoxInfo box;
			box.center = Vec3f(0.0f);
			Vec3f tmp = (texMesh->shapes()[it]->boundingBox.v1 - texMesh->shapes()[it]->boundingBox.v0) / 2;
			box.halfLength = Vec3f(abs(tmp.x), abs(tmp.y), abs(tmp.z));
			rigidbody.position = texMesh->shapes()[it]->boundingTransform.translation();
			Actors[it] = jeep->addBox(box, rigidbody, 100);
			jeep->bind(Actors[it], Pair<uint, uint>(it, 0));
		}
	}


	std::vector<int> bridge_id;
	for (int i = 5; i <= 51; ++i) {
		bridge_id.push_back(i);
	}

	for (auto it : bridge_id)
	{
		if (it == 5)
		{
			Vec3f offset = Vec3f(0.0f, 0.00, 0);
			rigidbody.offset = offset;
		}
		if (it == 6)
		{
			rigidbody.bodyId = 1;
			rigidbody.offset = Vec3f(0.0);
		}

		if (it == 51)
		{
			rigidbody.bodyId = 0;
		}
		auto up = texMesh->shapes()[it]->boundingBox.v1;
		auto down = texMesh->shapes()[it]->boundingBox.v0;
		BoxInfo box;
		box.center = Vec3f(0.0f);
		Vec3f tmp = (texMesh->shapes()[it]->boundingBox.v1 - texMesh->shapes()[it]->boundingBox.v0) / 2;

		rigidbody.position = texMesh->shapes()[it]->boundingTransform.translation();
		if (it <= 48)
		{
			box.halfLength = Vec3f(abs(tmp.x), abs(tmp.y), abs(tmp.z));
			Actors[it] = jeep->addBox(box, rigidbody, 100);
		}
		else if(it != 51)
		{
			box.halfLength = Vec3f(abs(tmp.x), abs(tmp.y) * 0.86, abs(tmp.z));
			Actors[it] = jeep->addBox(box, rigidbody, 10000000);
		}
		else
		{
			box.halfLength = Vec3f(abs(tmp.x), abs(tmp.y) * 0.1, abs(tmp.z));
			Actors[it] = jeep->addBox(box, rigidbody, 10);
		}

		
		jeep->bind(Actors[it], Pair<uint, uint>(it, 0));
	}

	for (auto it : bridge_id)
	{
		if (it <= 47 && it >= 6)
		{
			#ifdef USE_HINGE
				auto& hingeJoint = jeep->createHingeJoint(Actors[it], Actors[it + 1]);
				hingeJoint.setAnchorPoint((Actors[it]->center + Actors[it + 1]->center) / 2);
				hingeJoint.setAxis(Vec3f(1, 0, 0));
			#else
				auto& fixedJoint = jeep->createFixedJoint(Actors[it], Actors[it + 1]);
				fixedJoint.setAnchorPoint((Actors[it]->center + Actors[it + 1]->center) / 2);
			#endif 

		}
		if (it == 6)
		{
			auto& pointJoint = jeep->createHingeJoint(Actors[it], Actors[49]);
			pointJoint.setAnchorPoint(Actors[it]->center);
			pointJoint.setAxis(Vec3f(1, 0, 0));
		}
		if (it == 48)
		{
			auto& pointJoint = jeep->createHingeJoint(Actors[it], Actors[50]);
			pointJoint.setAnchorPoint(Actors[it]->center);
			pointJoint.setAxis(Vec3f(1, 0, 0));
		}
		if (it == 49)
		{
			auto& fixedJoint = jeep->createUnilateralFixedJoint(Actors[it+1]);
			fixedJoint.setAnchorPoint(Actors[it+1]->center);
			auto& fixedJoint2 = jeep->createUnilateralFixedJoint(Actors[it]);
			fixedJoint2.setAnchorPoint(Actors[it]->center);
			auto& fixedJoint3 = jeep->createUnilateralFixedJoint(Actors[it + 2]);
			fixedJoint3.setAnchorPoint(Actors[it + 2]->center);
			auto& fixedJoint4 = jeep->createFixedJoint(Actors[it], Actors[it + 1]);
			fixedJoint4.setAnchorPoint((Actors[it]->center + Actors[it + 1]->center) / 2);
		}
	}

	for (auto it : wheel_Id)
	{
		if (it != 4)
		{
			auto& hingeJoint = jeep->createHingeJoint(Actors[it], Actors[5]);
			hingeJoint.setAnchorPoint(Actors[it]->center);
			hingeJoint.setAxis(Vec3f(1, 0, 0));
			hingeJoint.setMoter(30);

		}
		else
		{
			auto& fixedJoint = jeep->createFixedJoint(Actors[it], Actors[5]);
			fixedJoint.setAnchorPoint((Actors[it]->center + Actors[5]->center) / 2);
		}
	}



	//Visualize rigid bodies
	
//  	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
//  	jeep->stateTopology()->connect(mapper->inDiscreteElements());
//  	jeep->graphicsPipeline()->pushModule(mapper);
//  
//  	auto sRender = std::make_shared<GLSurfaceVisualModule>();
//  	sRender->setColor(Color(0.3f, 0.5f, 0.9f));
//  	sRender->setAlpha(0.8f);
//  	sRender->setRoughness(0.7f);
//  	sRender->setMetallic(3.0f);
//  	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
// 	jeep->graphicsPipeline()->pushModule(sRender);

	//TODO: to enable using internal modules inside a node
	//Visualize contact normals

	return scn;
}

int main()
{
	UbiApp app(GUIType::GUI_GLFW);
	app.setSceneGraph(creatCar());
	app.initialize(1280, 768);

	//Set the distance unit for the camera, the fault unit is meter
	app.renderWindow()->getCamera()->setUnitScale(3.0f);

	app.mainLoop();

	return 0;
}


