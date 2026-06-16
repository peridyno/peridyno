#include <UbiApp.h>

#include <SceneGraph.h>

#include <RigidBody/Gear.h>
#include <RigidBody/MultibodySystem.h>
#include <RigidBody/Module/InstanceTransform.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLWireframeVisualModule.h>
#include <Module/GLPhotorealisticInstanceRender.h>

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Mapping/ContactsToEdgeSet.h>
#include <Mapping/ContactsToPointSet.h>

#include <BasicShapes/PlaneModel.h>

#include <Collision/NeighborElementQuery.h>

#include "RigidBody/Vehicle.h"

using namespace std;
using namespace dyno;


std::shared_ptr<SceneGraph> createSceneGraph()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setGravity(Vec3f(0.0f, -9.8f, 0.0f));


	auto plane = scn->addNode(std::make_shared<PlaneModel<DataType3f>>());
	plane->varLengthX()->setValue(160);
	plane->varLengthZ()->setValue(160);
	plane->varSegmentX()->setValue(160);
	plane->varSegmentZ()->setValue(160);

	auto convoy = scn->addNode(std::make_shared<MultibodySystem<DataType3f>>());
	convoy->varFrictionEnabled()->setValue(true);
	convoy->varSlop()->setValue(0.0015f);
	convoy->varSolverSubStepping()->setValue(12);
	convoy->varSolverIterationNumber()->setValue(30);
	convoy->varSolverHertz()->setValue(220.0f);
	convoy->varSolverDampingRatio()->setValue(1.0f);


	auto gear = scn->addNode(std::make_shared<MatBody<DataType3f>>());
	gear->varFrictionCoefficient()->setValue(0.7f);
	//gear->setXMLPath("ma/scene_chain_pyramid_impact_box_mass_x5.xml");
	//gear->setXMLPath("ma/scene_ragdoll_mesh_stack.xml");
	gear->setXMLPath("ma/stone_pile_1000.xml");
	//gear->setXMLPath("ma/scene_100_parkinglot.xml");
	auto meshTransformer = std::make_shared<InstanceTransform<DataType3f>>();
	gear->stateCenter()->connect(meshTransformer->inCenter());
	gear->stateRotationMatrix()->connect(meshTransformer->inRotationMatrix());
	gear->stateBindingPair()->connect(meshTransformer->inBindingPair());
	gear->stateBindingTag()->connect(meshTransformer->inBindingTag());
	gear->stateInstanceTransform()->connect(meshTransformer->inInstanceTransform());
	gear->graphicsPipeline()->pushModule(meshTransformer);

	auto meshRender = std::make_shared<GLPhotorealisticInstanceRender>();
	meshRender->setAlpha(1.0f);
	gear->stateTextureMesh()->connect(meshRender->inTextureMesh());
	meshTransformer->outInstanceTransform()->connect(meshRender->inTransform());
	gear->graphicsPipeline()->pushModule(meshRender);
	/*
	
	auto proxyMapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	gear->stateTopology()->connect(proxyMapper->inDiscreteElements());
	gear->graphicsPipeline()->pushModule(proxyMapper);

	auto proxyRender = std::make_shared<GLSurfaceVisualModule>();
	proxyRender->setColor(Color(0.0f, 0.8f, 0.7f));
	proxyRender->setAlpha(1.0f);
	proxyMapper->outTriangleSet()->connect(proxyRender->inTriangleSet());
	gear->graphicsPipeline()->pushModule(proxyRender);

	auto proxyWire = std::make_shared<GLWireframeVisualModule>();
	proxyWire->setColor(Color(0.0f, 0.0f, 0.0f));
	proxyMapper->outTriangleSet()->connect(proxyWire->inEdgeSet());
	gear->graphicsPipeline()->pushModule(proxyWire);
	*/
	gear->connect(convoy->importVehicles());

	//plane->stateTriangleSet()->connect(convoy->inTriangleSet());
	
	return scn;
}

int main()
{
	/*UbiApp app(GUIType::GUI_QT);
	app.setSceneGraph(createSceneGraph());

	app.initialize(1280, 768);
	app.renderWindow()->getCamera()->setUnitScale(3.0f);
	app.mainLoop();

	return 0;*/
	UbiApp app(GUIType::GUI_QT);
	app.setSceneGraph(createSceneGraph());
	app.initialize(1280, 768);
	app.mainLoop();

	return 0;
}
