#include <GlfwApp.h>
#include <QtGUI/QtApp.h>
#include <SceneGraph.h>
#include <Log.h>

#include <BasicShapes/CubeModel.h>

#include <Volume/BasicShapeToVolume.h>

#include <Multiphysics/VolumeBoundary.h>

#include <ParticleSystem/ParticleFluid.h>
#include "ParticleSystem/GhostParticles.h"
#include <ParticleSystem/GhostFluid.h>

#include <Module/CalculateNorm.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>

#include <ColorMapping.h>
#include <ImColorbar.h>



using namespace std;
using namespace dyno;

std::shared_ptr<ParticleSystem<DataType3f>> createFluidParticles()
{
	auto fluid = std::make_shared<ParticleSystem<DataType3f>>();

	std::vector<Vec3f> host_pos;
	std::vector<Vec3f> host_vel;

	Vec3f lo(-0.1, 0.0, -0.1);
	Vec3f hi(0.1f, 0.1f, 0.1f);

	Real s = 0.005f;
	int m_iExt = 0;

	float omega = 1.0f;
	float half_s = -s / 2.0f;

	int num = 0;

	for (float x = lo[0]; x <= hi[0]; x += s)
	{
		for (float y = lo[1]; y <= hi[1]; y += s)
		{
			for (float z = lo[2]; z <= hi[2]; z += s)
			{
				Vec3f p = Vec3f(x, y, z);
				host_pos.push_back(Vec3f(x, y, z));
				host_vel.push_back(Vec3f(0));
			}
		}
	}

	fluid->statePosition()->assign(host_pos);
	fluid->stateVelocity()->assign(host_vel);

	host_pos.clear();
	host_vel.clear();

	return fluid;
}

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
	scn->setUpperBound(Vec3f(0.5, 1, 0.5));
	scn->setLowerBound(Vec3f(-0.5, 0, -0.5));


	//Create a container
	auto cubeBoundary = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	cubeBoundary->varLocation()->setValue(Vec3f(0.0f, 0.5f, 0.0f));
	cubeBoundary->varLength()->setValue(Vec3f(1.0f));
	cubeBoundary->setVisible(false);

	auto cube2vol = scn->addNode(std::make_shared<BasicShapeToVolume<DataType3f>>());
	cube2vol->varGridSpacing()->setValue(0.02f);
	cube2vol->varInerted()->setValue(true);
	cubeBoundary->connect(cube2vol->importShape());

	auto boundary = scn->addNode(std::make_shared<VolumeBoundary<DataType3f>>());
	cube2vol->connect(boundary->importVolumes());

	auto fluid = scn->addNode(createFluidParticles());

	auto ghost = scn->addNode(createGhostParticles());

	auto incompressibleFluid = scn->addNode(std::make_shared<GhostFluid<DataType3f>>());
	incompressibleFluid->setDt(0.001f);
	fluid->connect(incompressibleFluid->importInitialStates());
	ghost->connect(incompressibleFluid->importBoundaryParticles());
// 	incompressibleFluid->setFluidParticles(fluid);
// 	incompressibleFluid->setBoundaryParticles(ghost);

// 	root->addAncestor(incompressibleFluid.get());
// 	root->addParticleSystem(fluid);
	incompressibleFluid->connect(boundary->importParticleSystems());

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
		auto ghostRender = std::make_shared<GLPointVisualModule>();
		ghostRender->setColor(Color(1, 0.5, 0));
		ghostRender->setColorMapMode(GLPointVisualModule::PER_OBJECT_SHADER);

		ghost->statePointSet()->connect(ghostRender->inPointSet());

		ghost->graphicsPipeline()->pushModule(ghostRender);
	}

	return scn;
}

int main()
{
	QtApp app;
	app.setSceneGraph(createScene());
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}


