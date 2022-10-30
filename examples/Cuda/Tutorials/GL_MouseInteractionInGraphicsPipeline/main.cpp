#include <GlfwApp.h>
#include <GLRenderEngine.h>

#include "Array/Array.h"
#include "Matrix.h"
#include "Node.h"

#include "SceneGraph.h"
#include "GLSurfaceVisualModule.h"
#include "CustomMouseInteraction.h"
#include "Algorithm/Reduction.h"

using namespace dyno;


class TransformNode : public Node
{
public:
	TransformNode() :
		topology("topology", "Topology", FieldTypeEnum::State, this) 
	{
		
	};

	std::shared_ptr<TriangleSet<DataType3f>> loadMesh(const std::string& path) {
		// load triangle mesh
		auto triSet = std::make_shared<TriangleSet<DataType3f>>();
		triSet->loadObjFile(getAssetPath() + path);
		this->setMesh(triSet);
		return triSet;
	}

	void setMesh(std::shared_ptr<TriangleSet<DataType3f>> triSet) {
		topology.setDataPtr(triSet);
		// update bounding box
		auto points = triSet->getPoints();
		Reduction<Vec3f> reduce;
		bbox.lower = reduce.minimum(points.begin(), points.size());
		bbox.upper = reduce.maximum(points.begin(), points.size());

	}

	void setTransform(const Transform3f& tm) {
		transform.assign(std::vector<Transform3f>{tm});
		transformCPU = tm;
	}

	void setSurfaceVisualModule(std::shared_ptr<GLSurfaceVisualModule> m) {
		topology.connect(m->inTriangleSet());
		transform.connect(m->inInstanceTransform());

		this->graphicsPipeline()->pushModule(m);
	}

	virtual NBoundingBox boundingBox() override {
		// simply transform the bounding box of original triangle set...
		NBoundingBox result(Vec3f(FLT_MAX), Vec3f(-FLT_MAX));

		Vec3f masks[8] = {
			{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
			{1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1},
		};

		for (Vec3f mask : masks) {
			Vec3f p = (Vec3f(1) - mask) * bbox.lower + mask * bbox.upper;
			p = transformCPU * p;
			result.join({ p, p });
		}

		return result;
	}

private:
	Transform3f								transformCPU;
	FArray<Transform3f, DeviceType::GPU>	transform;

	FInstance<TopologyModule>				topology;

	// bounding box of original triangle set
	NBoundingBox							bbox;
};


int main(int, char**)
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto triSet = std::make_shared<TriangleSet<DataType3f>>();
	triSet->loadObjFile(getAssetPath() + "armadillo/armadillo.obj");

	for (uint i = 0; i < 5; i++)
	{
		Transform3f tm;
		tm.translation() = Vec3f(0.4 * i, 0, 0);
		tm.scale() = Vec3f(1.0 + 0.1 * i, 1.0 - 0.1 * i, 1.0);
		tm.rotation() = Quat<float>(i * (-0.2), Vec3f(1, 0, 0)).toMatrix3x3();

		auto node = scn->addNode(std::make_shared<TransformNode>());
		node->setName("TN-" + std::to_string(i));
		//node->loadMesh("armadillo/armadillo.obj");
		node->setMesh(triSet);
		node->setTransform(tm);

		auto sm = std::make_shared<GLSurfaceVisualModule>();
		sm->setColor(Vec3f(i * 0.2f, i * 0.2f, 1.f - i * 0.1f));
		sm->setAlpha(0.8f);

		node->setSurfaceVisualModule(sm);
	}

	scn->setUpperBound({ 4, 4, 4 });

	GlfwApp window;
	window.setSceneGraph(scn);
	window.createWindow(1024, 768);
	window.mainLoop();

	return 0;
}
