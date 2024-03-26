#include <QtApp.h>

#include "initializeModeling.h"
#include "PointFromCurve.h"
#include "ObjIO/OBJexporter.h"
#include "Sweep.h"
#include "Extrude.h"
#include "Curve.h"
#include "Ramp.h"
#include "Canvas.h"
#include "GltfLoader.h"
#include "GLPhotorealisticRender.h"

#include "GLWireframeVisualModule.h"
#include "Matrix/SquareMatrix.h"
#include "Matrix/Matrix4x4.h"
#include "GraphicsObject/Instance.h"
#include "GLInstanceVisualModule.h"
#include "Node/GLInstanceVisualNode.h"

using namespace dyno;

class Instances : public Node
{
public:
	Instances() {

		
	};

	void resetStates() override 
	{
		if (this->inWorldMatrix()->isEmpty())
			return;

		auto jointMatrix = this->inWorldMatrix()->getData();

		int copyNum = 5;
		int shapeNum = jointMatrix.size();

		CArray<Transform3f> cT;
		CArray<Mat4f> cM;
		Mat4f tempM = Mat4f::identityMatrix();

		for (size_t i = 0; i < copyNum; i++)
		{
			for (size_t j = 0; j < shapeNum; j++)
			{
				cT.pushBack(Transform3f(Vec3f(i, 0, 0), Mat3f::identityMatrix(), Vec3f(1, 1, 1)));
				tempM(0, 3) = i;
				cM.pushBack(tempM);
			}
		}
		this->stateTransform()->assign(cT);
		this->stateMatrix()->assign(cM);

		this->stateShapeInstances()->resize(shapeNum);
		auto& stateInstances = this->stateShapeInstances()->getData();

		std::vector<std::vector<Transform3f>> transform;
		transform.resize(shapeNum);
		for (size_t i = 0; i < shapeNum; i++)
		{
			for (size_t j = 0; j < copyNum; j++)
			{
				transform[i].push_back(Transform3f(Vec3f(j, 0, 0), Mat3f::identityMatrix(), Vec3f(1, 1, 1)));
				std::cout << transform[i][transform[i].size() - 1].translation().x << ", " << transform[i][transform[i].size() - 1].translation().y << ", " << transform[i][transform[i].size() - 1].translation().z << std::endl;
			
				
			}
			copyNum++;

		}

		for (size_t i = 0; i < stateInstances.size(); i++)
		{
			stateInstances[i] = std::make_shared<ShapeInstance>();
			stateInstances[i]->transform.assign(transform[i]);
			stateInstances[i]->instanceCount = transform[i].size();
		}

		auto test = this->stateShapeInstances()->getData();
		for (size_t i = 0; i < test.size(); i++)
		{
			std::cout << "shape : "<< i << " : " << test[i]->transform.size() << std::endl;
		}

	}


	void coutState() 
	{
		CArray<Mat4f> cm;
		CArray<Transform3f> ct;

		cm.assign(this->stateMatrix()->getData());
		ct.assign(this->stateTransform()->getData());

		for (size_t i = 0; i < cm.size(); i++)
		{
			std::cout << cm[i](0, 3) << ", " << cm[i](1, 3) << ", " << cm[i](2, 3) << std::endl;
			
		}

		for (size_t i = 0; i < ct.size(); i++)
		{
			std::cout << ct[i].translation().x << ", " << ct[i].translation().y << ", " << ct[i].translation().z << ", " << std::endl;

		}


	}


	DEF_ARRAY_IN(Mat4f, WorldMatrix, DeviceType::GPU, "CoordChannel_1");

	DEF_ARRAY_STATE(Mat4f, Matrix, DeviceType::GPU, "CoordChannel_1");
	DEF_ARRAY_STATE(Transform3f, Transform, DeviceType::GPU, "CoordChannel_1");

	DEF_ARRAYLIST_STATE(Mat4f, MatrixList, DeviceType::GPU, "CoordChannel_1");
	DEF_ARRAYLIST_STATE(Transform3f, TransformList, DeviceType::GPU, "CoordChannel_1");

	DEF_INSTANCES_STATE(ShapeInstance, ShapeInstance, "");
};


using namespace dyno;

int main()
{
	//Create SceneGraph
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto gltf = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	gltf->varFileName()->setValue(std::string("C:/Users/dell/Desktop/testcube_noTex.gltf"));
	
	auto module = gltf->graphicsPipeline()->findFirstModule<GLWireframeVisualModule>();
	//gltf->deleteModule(module);
	gltf->graphicsPipeline()->clear();
	gltf->graphicsPipeline()->pushModule(module);

	auto instanceData = scn->addNode(std::make_shared<Instances>());
	gltf->stateJointWorldMatrix()->connect(instanceData->inWorldMatrix());

	auto instanceVisualNode = scn->addNode(std::make_shared<GLInstanceVisualNode<DataType3f>>());
	gltf->stateShapes()->connect(instanceVisualNode->inShapes());
	gltf->stateMaterials()->connect(instanceVisualNode->inMaterials());
	gltf->stateNormal()->connect(instanceVisualNode->inNormal());
	gltf->stateTexCoord_0()->connect(instanceVisualNode->inTexCoord());
	gltf->stateVertex()->connect(instanceVisualNode->inVertex());
	instanceData->stateShapeInstances()->connect(instanceVisualNode->inInstances());




	Modeling::initStaticPlugin();


	QtApp app;
	app.setSceneGraph(scn);
	app.initialize(1366, 800);
	app.mainLoop();

	return 0;
}