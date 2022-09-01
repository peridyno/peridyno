#include <QtApp.h>
#include <GlfwApp.h>

#include "Array/Array.h"
#include "Matrix.h"
#include "Node.h"
#include <math.h>
#include "SceneGraph.h"
#include "GLSurfaceVisualModule.h"
#include "GLInstanceVisualModule.h"

#include "ParticleSystem/Module/ParticleIntegrator.h"

#include <RigidBody/RigidBodySystem.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <GLWireframeVisualModule.h>
#include <Mapping/ContactsToEdgeSet.h>
#include "Collision/NeighborElementQuery.h"

#include <Mapping/DiscreteElementsToTriangleSet.h>


#include "Camera.h"

#define PI 3.14159265
using namespace dyno;

class Instances : public Node
{
public:
	float theata = 0;

	Instances() {
		Transform3f tm;
		CArray<Transform3f> hTransform;
		for (uint i = 0; i < 1; i++)
		{
			//tm.translation() = Vec3f(0.4 * i, 0, 0);
			//tm.scale() = Vec3f(1.0 + 0.1 * i, 1.0 - 0.1 * i, 1.0);
			tm.rotation() = Quat<float>(i * (-0.2), Vec3f(1, 0, 0)).toMatrix3x3();
			hTransform.pushBack(tm);
		}

		this->stateTransforms()->allocate()->assign(hTransform);

		std::shared_ptr<TriangleSet<DataType3f>> triSet = std::make_shared<TriangleSet<DataType3f>>();
		triSet->loadObjFile(getAssetPath() + "cloth/cloth.obj");

		this->stateTopology()->setDataPtr(triSet);

		auto ptSet = TypeInfo::cast<TriangleSet<DataType3f>>(this->stateTopology()->getDataPtr());
		if (ptSet == nullptr) return;

		auto pts = ptSet->getPoints();

		if (pts.size() > 0)
		{
			this->statePosition()->resize(pts.size());
			this->statePosition()->assign(pts);
		}

		auto integrator = std::make_shared<ParticleIntegrator<DataType3f>>();
		
		this->statePosition()->connect(integrator->inPosition());

		hTransform.clear();
	};

	void Instances::resetStates(){

		auto& mPoints = this->statePosition()->getData();
		CArray<Vec3f> cPoints;
		cPoints.resize(mPoints.size());
		cPoints.assign(mPoints);
		for (int i = 0; i < cPoints.size(); i++) {
			cPoints[i].y = 0;
			cPoints[i].x -= 0.5;
		}
		mPoints.assign(cPoints);
		
		this->updateTopology();
	}

	void Instances::updateStates() {

		auto& mPoints = this->statePosition()->getData();

		CArray<Vec3f> cPoints;
		cPoints.resize(mPoints.size());
		cPoints.assign(mPoints);
		

		float n = theata * PI / 180.0;
		for (int i = 0; i < cPoints.size(); i++) {	
			cPoints[i] = Mat3f(cos(n), -sin(n), 0,
				sin(n), cos(n), 0,
				0, 0, 1) * cPoints[i];
		}
		theata= 0.001+theata;
		

		mPoints.assign(cPoints);
		
	}


	void Instances::updateTopology()
	{
		auto triSet = TypeInfo::cast<TriangleSet<DataType3f>>(this->stateTopology()->getDataPtr());

		triSet->getPoints().assign(this->statePosition()->getData());
	}

	DEF_ARRAY_STATE(Vec3f, Position, DeviceType::GPU, "position");
	DEF_ARRAY_STATE(Transform3f, Transforms, DeviceType::GPU, "Instance transform");

	DEF_INSTANCE_STATE(TopologyModule, Topology, "Topology");
};

int main(int, char**)
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto instanceNode = scn->addNode(std::make_shared<Instances>());

	auto surfaceRenderer = std::make_shared<GLSurfaceVisualModule>();

	//surfaceRenderer->setColor(Vec3f(0, 1, 0));
	//instanceNode->stateTopology()->connect(surfaceRenderer->inTriangleSet());
	//instanceNode->graphicsPipeline()->pushModule(surfaceRenderer);

	auto rigid = scn->addNode(std::make_shared<RigidBodySystem<DataType3f>>());
	RigidBodyInfo rigidBody;
	rigidBody.linearVelocity = Vec3f(0, 0, 0);
	BoxInfo box;

	box.center = 0.5f * Vec3f(0, 0.4, 0);
	box.halfLength = Vec3f(1.0, 1.0, 1.0);
	rigid->addBox(box, rigidBody);

	scn->setLowerBound(box.center - box.halfLength);
	scn->setUpperBound(box.center + box.halfLength);

	auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
	rigid->stateTopology()->connect(mapper->inDiscreteElements());
	rigid->graphicsPipeline()->pushModule(mapper);

	auto sRender = std::make_shared<GLSurfaceVisualModule>();
	sRender->setColor(Vec3f(1, 1, 0));
	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
	rigid->graphicsPipeline()->pushModule(sRender);

	scn->addNode(rigid);

	GlfwApp window;
	
	//Set the distance unit for the camera, the fault unit is meter
	window.activeCamera()->setDistanceUnit(3.0f);
	window.setSceneGraph(scn);
	window.createWindow(1024, 768);

	window.mainLoop();

	return 0;
}
