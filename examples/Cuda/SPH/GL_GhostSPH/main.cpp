#include <UbiApp.h>
#include <SceneGraph.h>
#include <Log.h>

#include <BasicShapes/CubeModel.h>
#include <Volume/BasicShapeToVolume.h>
#include <Multiphysics/VolumeBoundary.h>

#include <ParticleSystem/ParticleFluid.h>
#include <ParticleSystem/GhostParticles.h>
#include <ParticleSystem/GhostFluid.h>

#include <Module/CalculateNorm.h>
#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>
#include <ImColorbar.h>



using namespace std;
using namespace dyno;
std::shared_ptr<ParticleSystem<DataType3f>> createFluidParticles(Vec3f lo, Vec3f hi)
{
	auto fluid = std::make_shared<ParticleSystem<DataType3f>>();

	std::vector<Vec3f> host_pos;
	std::vector<Vec3f> host_vel;



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

std::shared_ptr<GhostParticles<DataType3f>> createGhostParticles(Vec3f low, Vec3f high)
{
	auto ghost = std::make_shared<GhostParticles<DataType3f>>();

	std::vector<Vec3f> host_pos;
	std::vector<Vec3f> host_vel;
	std::vector<Vec3f> host_force;
	std::vector<Vec3f> host_normal;
	std::vector<Attribute> host_attribute;


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
	cubeBoundary->varLength()->setValue(Vec3f(0.4f, 1.0f, 0.4f));

	auto cube2vol = scn->addNode(std::make_shared<BasicShapeToVolume<DataType3f>>());
	cube2vol->varGridSpacing()->setValue(0.02f);
	cube2vol->varInerted()->setValue(true);
	cubeBoundary->connect(cube2vol->importShape());

	auto boundary = scn->addNode(std::make_shared<VolumeBoundary<DataType3f>>());
	cube2vol->connect(boundary->importVolumes());

	auto fluid = scn->addNode(createFluidParticles(Vec3f(-0.1, 0.2, -0.1),Vec3f(0.1f, 0.25f, 0.1f)));
	auto ghost1 = scn->addNode(createGhostParticles(Vec3f(-0.2, -0.015, -0.2), Vec3f(0.2, -0.005, 0.2)));
	auto ghost2 = scn->addNode(createGhostParticles(Vec3f(-0.1, 0.1, -0.1), Vec3f(0.1, 0.125, 0.1)));

	auto incompressibleFluid = scn->addNode(std::make_shared<GhostFluid<DataType3f>>());
	incompressibleFluid->setDt(0.001f);
	incompressibleFluid->varSmoothingLength()->setValue(2.5f);
	fluid->connect(incompressibleFluid->importInitialStates());
	ghost1->connect(incompressibleFluid->importBoundaryParticles());
	ghost2->connect(incompressibleFluid->importBoundaryParticles());
	incompressibleFluid->connect(boundary->importParticleSystems());

	return scn;
}

int main()
{
	UbiApp app(GUIType::GUI_QT);
	app.setSceneGraph(createScene());
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}


