//#include <GlfwApp.h>
#include <QtApp.h>
#include <SceneGraph.h>
#include <Log.h>

#include <ParticleSystem/ParticleFluid.h>
#include "ParticleSystem/GhostParticles.h"
#include "ParticleSystem/MakeGhostParticles.h"
#include <ParticleSystem/StaticBoundary.h>
#include <ParticleSystem/GhostFluid.h>
#include <ParticleSystem/CubeSampler.h>
#include "ParticleSystem/MakeParticleSystem.h"

#include <Module/CalculateNorm.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>

#include <ColorMapping.h>
#include <ImColorbar.h>

#include <BasicShapes/CubeModel.h>
#include <BasicShapes/SphereModel.h>
#include <BasicShapes/PlaneModel.h>
#include "Commands/PointsBehindMesh.h"
#include "SemiAnalyticalScheme/ParticleRelaxtionOnMesh.h"
#include "ObjIO/ObjLoader.h"
#include "GLSurfaceVisualModule.h"

using namespace std;
using namespace dyno;

bool useVTK = false;

std::shared_ptr<GhostParticles<DataType3f>> createGhostParticles()
{
	auto ghost = std::make_shared<GhostParticles<DataType3f>>();

	std::vector<Vec3f> host_pos;
	std::vector<Vec3f> host_vel;
	std::vector<Vec3f> host_force;
	std::vector<Vec3f> host_normal;
	std::vector<Attribute> host_attribute;

	Vec3f low(-0.2, -0.015, -0.2);
	Vec3f high(0.2, -0.005, 0.2);

	Real s = 0.005f;
	int m_iExt = 0;

	float omega = 1.0f;
	float half_s = -s / 2.0f;

	int num = 0;

	for (float x = low.x - m_iExt * s; x <= high.x + m_iExt * s; x += s) {
		for (float y = low.y - m_iExt * s; y <= high.y + m_iExt * s; y += s) {
			for (float z = low.z - m_iExt * s; z <= high.z + m_iExt * s; z += s) {
				Attribute attri;
				attri.setFluid();
				attri.setDynamic();

				host_pos.push_back(Vec3f(x, y, z));
				host_vel.push_back(Vec3f(0));
				host_force.push_back(Vec3f(0));
				host_normal.push_back(Vec3f(0, 1, 0));
				host_attribute.push_back(attri);
			}
		}
	}

	ghost->statePosition()->resize(num);
	ghost->stateVelocity()->resize(num);

	ghost->stateNormal()->resize(num);
	ghost->stateAttribute()->resize(num);

	ghost->statePosition()->assign(host_pos);
	ghost->stateVelocity()->assign(host_vel);
	ghost->stateNormal()->assign(host_normal);
	ghost->stateAttribute()->assign(host_attribute);

	host_pos.clear();
	host_vel.clear();
	host_force.clear();
	host_normal.clear();
	host_attribute.clear();

	return ghost;
}

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(10.5, 5, 10.5));
	scn->setLowerBound(Vec3f(-10.5, -5, -10.5));

	auto obj1 = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	obj1->varScale()->setValue(Vec3f(0.3));
	obj1->varFileName()->setValue(getAssetPath() + "plane/plane_lowRes.obj");
	//obj1->varFileName()->setValue(getAssetPath() + "board/ball.obj");
	obj1->varLocation()->setValue(Vec3f(0.0, 0.0, 0.0));
	auto SurfaceModule1 = obj1->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	SurfaceModule1->setColor(Color(0.2));
	SurfaceModule1->setMetallic(1);
	SurfaceModule1->setRoughness(0.8);

	//auto pointset_1 = scn->addNode(std::make_shared<PointsBehindMesh<DataType3f>>());
	//pointset_1->varGeneratingDirection()->setValue(false);
	//pointset_1->varSamplingDistance()->setValue(0.005);
	//pointset_1->varThickness()->setValue(0.045);
	//obj1->outTriangleSet()->connect(pointset_1->inTriangleSet());

	/*Generate points on meshes of the cube, and relax the point positions*/
	auto pointset_1 = scn->addNode(std::make_shared<ParticleRelaxtionOnMesh<DataType3f>>());
	pointset_1->varSamplingDistance()->setValue(0.005);
	pointset_1->varThickness()->setValue(0.045);
	obj1->outTriangleSet()->connect(pointset_1->inTriangleSet());
	pointset_1->graphicsPipeline()->clear();

	//auto pointset_1 = scn->addNode(std::make_shared<PointsBehindMesh<DataType3f>>());
	//pointset_1->varSamplingDistance()->setValue(0.005);
	//pointset_1->varThickness()->setValue(0.045);
	//obj1->outTriangleSet()->connect(pointset_1->inTriangleSet());
	//pointset_1->graphicsPipeline()->clear();

	auto ghost2 = scn->addNode(std::make_shared<MakeGhostParticles<DataType3f>>());
	pointset_1->statePointSet()->connect(ghost2->inPoints());
	pointset_1->statePointNormal()->connect(ghost2->stateNormal());

//	auto ghost2 = scn->addNode(createGhostParticles());


	//Create a cube
	auto cube = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube->varLocation()->setValue(Vec3f(0.0, 0.3, 0.0));
	cube->varLength()->setValue(Vec3f(0.2, 0.2, 0.2));
	cube->graphicsPipeline()->disable();

	
	auto sampler = scn->addNode(std::make_shared<CubeSampler<DataType3f>>());
	sampler->varSamplingDistance()->setValue(0.005);
	sampler->setVisible(false);

	cube->outCube()->connect(sampler->inCube());

	auto fluidParticles = scn->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());

	sampler->statePointSet()->promoteOuput()->connect(fluidParticles->inPoints());

	auto incompressibleFluid = scn->addNode(std::make_shared<GhostFluid<DataType3f>>());
	fluidParticles->connect(incompressibleFluid->importFluidParticles());
	ghost2->connect(incompressibleFluid->importBoundaryParticles());

	{
		auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
		auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
		colorMapper->varMax()->setValue(5.0f);

		incompressibleFluid->stateVelocity()->connect(calculateNorm->inVec());
		calculateNorm->outNorm()->connect(colorMapper->inScalar());

		incompressibleFluid->graphicsPipeline()->pushModule(calculateNorm);
		incompressibleFluid->graphicsPipeline()->pushModule(colorMapper);

		auto ptRender = std::make_shared<GLPointVisualModule>();
		ptRender->setColor(Color(1, 0, 0));
		ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);

		incompressibleFluid->statePointSet()->connect(ptRender->inPointSet());
		colorMapper->outColor()->connect(ptRender->inColor());

		incompressibleFluid->graphicsPipeline()->pushModule(ptRender);
	}


	{
		auto ptRender = std::make_shared<GLPointVisualModule>();
		ptRender->setColor(Color(0.6, 0.5, 0.2));
		ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
		pointset_1->statePointSet()->connect(ptRender->inPointSet());
		pointset_1->graphicsPipeline()->pushModule(ptRender);
	}
	return scn;
}

int main()
{
	//GlfwApp app;
	QtApp app;
	app.setSceneGraph(createScene());
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}


