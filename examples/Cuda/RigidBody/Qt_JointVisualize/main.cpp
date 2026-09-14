#include <QtApp.h>

#include <SceneGraph.h>
#include <HeightField/GranularMedia.h>
#include <BasicShapes/PlaneModel.h>
#include <BasicShapes/PlaneModel.h>

#include <RigidBody/ConfigurableBody.h>
#include <RigidBody/Module/CarDriver.h>

#include "BasicShapes/PlaneModel.h"
#include "FBXLoader/FBXLoader.h"
#include "RigidBody/Module/AnimationDriver.h"
#include "RigidBody/MultibodySystem.h"
#include <HeightField/SurfaceParticleTracking.h>
#include <HeightField/RigidSandCoupling.h>
#include "GLWireframeVisualModule.h"
#include "GLInstanceVisualModule.h"
#include "Mapping/DiscreteElementsJointToInstance.h"
#include "Mapping/DiscreteElementsToTriangleSet.h"
#include "GLDiscreteElementVisualModule.h"
#include "GLDiscreteElementJointVisualModule.h"

using namespace std;
using namespace dyno;

#define ELEMENTALPHA 0.5f
#define JOINTALPHA 1.0f
#define THICKNESSSCALE 1.0f


std::shared_ptr<SceneGraph> creatCar()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	bool useVisual = true;

	{
		auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());

		RigidBodyInfo rigidBody;
		rigidBody.friction = 0.01;
		BoxInfo box1, box2;

		box1.halfLength = Vec3f(0.09, 0.1, 0.1);

		rigidBody.position = Vec3f(0, 0.5, 0);

		auto boxActor1 = rigid->addBox(box1, rigidBody, 1000);

		box2.halfLength = Vec3f(0.1, 0.4, 0.2);

		rigidBody.position = Vec3f(0.2, 0.5, 0);
		rigidBody.angularVelocity = Vec3f(1, 0, 0);
		rigidBody.motionType = BodyType::Kinematic;

		auto boxActor2 = rigid->addBox(box2, rigidBody);

		auto& sliderJoint = rigid->createSliderJoint(boxActor1, boxActor2);

		sliderJoint.setAnchorPoint((boxActor1->center + boxActor2->center) / 2);
		sliderJoint.setAxis(Vec3f(0, 1, 0));
		sliderJoint.setRange(-0.2, 0.2);

		if (useVisual)
		{
			auto elementVisualize = std::make_shared<GLDiscreteElementVisualModule<DataType3f>>();
			rigid->stateTopology()->connect(elementVisualize->inDiscreteElements());
			rigid->graphicsPipeline()->pushModule(elementVisualize);
			elementVisualize->varAlpha()->setValue(ELEMENTALPHA);

			auto jointVisualize = std::make_shared<GLDiscreteElementJointVisualModule<DataType3f>>();
			rigid->stateTopology()->connect(jointVisualize->inDiscreteElements());
			rigid->graphicsPipeline()->pushModule(jointVisualize);
			jointVisualize->varAlpha()->setValue(JOINTALPHA);
			jointVisualize->varRenderToFront()->setValue(true);
			jointVisualize->varReceiveShadow()->setValue(false);
			jointVisualize->varThicknessScale()->setValue(THICKNESSSCALE * 0.8f);
		}
	}
	
	{
		auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());

		BoxInfo box;
		RigidBodyInfo rA;
		rA.bodyId = 1;
		rA.linearVelocity = Vec3f(1, 0.0, 0.0);
		rA.position = Vec3f(-0.5f,0,0);
		box.center = Vec3f(0, 0.0, 0);
		box.halfLength = Vec3f(0.05, 0.05, 0.05);
		auto oldBoxActor = rigid->addBox(box, rA);

		for (int i = 0; i < 5; i++)
		{
			RigidBodyInfo rB;
			rB.position = rA.position + Vec3f(0.0, 0.12f, 0.0);
			rB.linearVelocity = Vec3f(0, 0, 0);

			auto newBoxActor = rigid->addBox(box, rB);
			auto& ballAndSocketJoint = rigid->createBallAndSocketJoint(oldBoxActor, newBoxActor);
			ballAndSocketJoint.setAnchorPoint((rA.position + rB.position) / 2);

			rA = rB;
			oldBoxActor = newBoxActor;
		}
		if (useVisual)
		{
			auto elementVisualize = std::make_shared<GLDiscreteElementVisualModule<DataType3f>>();
			rigid->stateTopology()->connect(elementVisualize->inDiscreteElements());
			rigid->graphicsPipeline()->pushModule(elementVisualize);
			elementVisualize->varAlpha()->setValue(ELEMENTALPHA);

			auto jointVisualize = std::make_shared<GLDiscreteElementJointVisualModule<DataType3f>>();
			rigid->stateTopology()->connect(jointVisualize->inDiscreteElements());
			rigid->graphicsPipeline()->pushModule(jointVisualize);
			jointVisualize->varAlpha()->setValue(JOINTALPHA);
			jointVisualize->varRenderToFront()->setValue(true);
			jointVisualize->varReceiveShadow()->setValue(false);
			jointVisualize->varThicknessScale()->setValue(THICKNESSSCALE * 0.20f);
		}
	}

	{
		auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());

		BoxInfo box;
		box.center = Vec3f(0.0f);
		box.halfLength = Vec3f(0.02, 0.02, 0.02);

		RigidBodyInfo rbA;
		RigidBodyInfo rbB;
		rbA.position = Vec3f(0.5f, 0.1f, 0.0f);
		rbA.linearVelocity = Vec3f(1.0, 0.0, 1.0);

		auto oldBoxActor = rigid->addBox(box, rbA);

		rbA.linearVelocity = Vec3f(0, 0, 0);

		for (int i = 1; i < 5; i++)
		{
			rbB.position = rbA.position + Vec3f(0, 0.05f, 0.0);
			rbB.angle = Quat1f(M_PI / 3 * i, Vec3f(0, 1, 0));
			auto newBoxActor = rigid->addBox(box, rbB);

			auto& fixedJoint = rigid->createFixedJoint(oldBoxActor, newBoxActor);
			fixedJoint.setAnchorPoint((rbA.position + rbB.position) / 2);


			rbA = rbB;
			oldBoxActor = newBoxActor;
		}
		if (useVisual)
		{
			auto elementVisualize = std::make_shared<GLDiscreteElementVisualModule<DataType3f>>();
			rigid->stateTopology()->connect(elementVisualize->inDiscreteElements());
			rigid->graphicsPipeline()->pushModule(elementVisualize);
			elementVisualize->varAlpha()->setValue(ELEMENTALPHA);

			auto jointVisualize = std::make_shared<GLDiscreteElementJointVisualModule<DataType3f>>();
			rigid->stateTopology()->connect(jointVisualize->inDiscreteElements());
			rigid->graphicsPipeline()->pushModule(jointVisualize);
			jointVisualize->varAlpha()->setValue(JOINTALPHA);
			jointVisualize->varRenderToFront()->setValue(true);
			jointVisualize->varReceiveShadow()->setValue(false);
			jointVisualize->varThicknessScale()->setValue(THICKNESSSCALE * 1.0f);
		}

	}

	{
		auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());

		BoxInfo box;
		box.center = Vec3f(0.0f);
		box.halfLength = Vec3f(0.02, 0.02, 0.02);

		RigidBodyInfo rbA;
		RigidBodyInfo rbB;
		rbA.position = Vec3f(1.0f, 0.1f, 0.0f);
		rbA.linearVelocity = Vec3f(1.0, 0.0, 1.0);

		auto oldBoxActor = rigid->addBox(box, rbA);

		rbA.linearVelocity = Vec3f(0, 0, 0);

		for (int i = 1; i < 5; i++)
		{
			rbB.position = rbA.position + Vec3f(0, 0.05f, 0.0);
			rbB.angle = Quat1f(M_PI / 3 * i, Vec3f(0, 1, 0));
			auto newBoxActor = rigid->addBox(box, rbB);

			auto& fixedJoint = rigid->createPointJoint(oldBoxActor);
			fixedJoint.setAnchorPoint((rbA.position + rbB.position) / 2);


			rbA = rbB;
			oldBoxActor = newBoxActor;
		}
		if (useVisual)
		{
			auto elementVisualize = std::make_shared<GLDiscreteElementVisualModule<DataType3f>>();
			rigid->stateTopology()->connect(elementVisualize->inDiscreteElements());
			rigid->graphicsPipeline()->pushModule(elementVisualize);
			elementVisualize->varAlpha()->setValue(ELEMENTALPHA);

			auto jointVisualize = std::make_shared<GLDiscreteElementJointVisualModule<DataType3f>>();
			rigid->stateTopology()->connect(jointVisualize->inDiscreteElements());
			rigid->graphicsPipeline()->pushModule(jointVisualize);
			jointVisualize->varAlpha()->setValue(JOINTALPHA);
			jointVisualize->varRenderToFront()->setValue(true);
			jointVisualize->varReceiveShadow()->setValue(false);
			jointVisualize->varThicknessScale()->setValue(THICKNESSSCALE * 0.2f);
		}
	}


	{
		auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());

		RigidBodyInfo rA;
		BoxInfo box;
		box.center = Vec3f(0.0f);
		box.halfLength = Vec3f(0.04, 0.04, 0.04);
		rA.position = Vec3f(-1.0f, 0, 0);
		rA.linearVelocity = Vec3f(1, 0, 0);
		auto oldBoxActor = rigid->addBox(box, rA);
		rA.linearVelocity = Vec3f(0, 0, 0);

		for (int i = 0; i < 5; i++)
		{
			RigidBodyInfo rB;
			rB.position = rA.position + 2.0 * Vec3f(0.0, 0.1, 0);
			rB.bodyId = i + 1;

			auto newBoxActor = rigid->addBox(box, rB);
			auto& hingeJoint = rigid->createHingeJoint(oldBoxActor, newBoxActor);
			hingeJoint.setAnchorPoint((rA.position + rB.position) / 2);
			hingeJoint.setAxis(Vec3f(0, 0, 1));
			hingeJoint.setRange(-M_PI / 2, M_PI / 2);

			rA = rB;
			oldBoxActor = newBoxActor;
		}
		if (useVisual)
		{
			auto elementVisualize = std::make_shared<GLDiscreteElementVisualModule<DataType3f>>();
			rigid->stateTopology()->connect(elementVisualize->inDiscreteElements());
			rigid->graphicsPipeline()->pushModule(elementVisualize);
			elementVisualize->varAlpha()->setValue(ELEMENTALPHA);

			auto jointVisualize = std::make_shared<GLDiscreteElementJointVisualModule<DataType3f>>();
			rigid->stateTopology()->connect(jointVisualize->inDiscreteElements());
			rigid->graphicsPipeline()->pushModule(jointVisualize);
			jointVisualize->varAlpha()->setValue(JOINTALPHA);
			jointVisualize->varRenderToFront()->setValue(true);
			jointVisualize->varReceiveShadow()->setValue(false);
			jointVisualize->varThicknessScale()->setValue(THICKNESSSCALE * 0.3f);
		}
	}

	{
		auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());

		RigidBodyInfo rA;
		BoxInfo box;
		box.center = Vec3f(0.0f);
		box.halfLength = Vec3f(0.04, 0.04, 0.04);
		rA.position = Vec3f(-1.5f, 0, 0);
		rA.linearVelocity = Vec3f(1, 0, 0);
		auto oldBoxActor = rigid->addBox(box, rA);
		rA.linearVelocity = Vec3f(0, 0, 0);

		for (int i = 0; i < 5; i++)
		{
			RigidBodyInfo rB;
			rB.position = rA.position + 2.0 * Vec3f(0.0, 0.1, 0);
			rB.bodyId = i + 1;

			auto newBoxActor = rigid->addBox(box, rB);
			auto& hingeJoint = rigid->createHingeJoint(oldBoxActor, newBoxActor);
			hingeJoint.setAnchorPoint((rA.position + rB.position) / 2);
			hingeJoint.setAxis(Vec3f(0, 0, 1));
			hingeJoint.setRange(-M_PI / 2, M_PI / 2);

			rA = rB;
			oldBoxActor = newBoxActor;
		}
		if (useVisual)
		{
			auto elementVisualize = std::make_shared<GLDiscreteElementVisualModule<DataType3f>>();
			rigid->stateTopology()->connect(elementVisualize->inDiscreteElements());
			rigid->graphicsPipeline()->pushModule(elementVisualize);
			elementVisualize->varAlpha()->setValue(ELEMENTALPHA);

			auto jointVisualize = std::make_shared<GLDiscreteElementJointVisualModule<DataType3f>>();
			rigid->stateTopology()->connect(jointVisualize->inDiscreteElements());
			rigid->graphicsPipeline()->pushModule(jointVisualize);
			jointVisualize->varAlpha()->setValue(JOINTALPHA);
			jointVisualize->varRenderToFront()->setValue(true);
			jointVisualize->varReceiveShadow()->setValue(false);
			jointVisualize->varThicknessScale()->setValue(THICKNESSSCALE * 0.3f);
		}
	}


	return scn;
}

int main()
{
	QtApp app;
	app.setSceneGraph(creatCar());
	app.initialize(1280, 768);

	//Set the distance unit for the camera, the fault unit is meter
	app.renderWindow()->getCamera()->setUnitScale(3.0f);

	app.mainLoop();

	return 0;
}


