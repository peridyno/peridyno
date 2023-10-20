#include <GlfwApp.h>
#include "QtGUI/QtApp.h"
#include <SceneGraph.h>
#include <Log.h>

#include <ParticleSystem/ParticleFluid.h>
#include "ParticleSystem/GhostParticles.h"
#include <ParticleSystem/StaticBoundary.h>
#include <ParticleSystem/GhostFluid.h>

#include <Module/CalculateNorm.h>

#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>

#include <ColorMapping.h>
#include <ImColorbar.h>
#include "ImChart.h"
#include "TriangleMeshWriter.h"
#include "GLPointVisualModule.h"



using namespace std;
using namespace dyno;

bool useVTK = false;

class PointFromVector : public Node
{
public:

	typedef typename DataType3f::Real Real;
	typedef typename DataType3f::Coord Coord;

	PointFromVector() 	
	{

		this->statePointSet()->setDataPtr(std::make_shared<PointSet<DataType3f>>());

		auto glp = std::make_shared<GLPointVisualModule>();
		this->statePointSet()->connect(glp->inPointSet());
		glp->varPointSize()->setValue(0.02);

		this->graphicsPipeline()->pushModule(glp);

	};

	void modifyPoints()
	{
		auto ptSet = this->statePointSet()->getDataPtr();

		std::vector<Coord> ptCoord;
		CArray<Real> cy;
		auto et = this->stateFrameNumber()->getValue();
		for (size_t i = 0; i < 50; i++)
		{
			ptCoord.push_back(Coord(float(i) /5,sin(float(i) +float(et)/5)/4,0));
			cy.pushBack(Real(sin(i+et)));
		}
		ptSet->setPoints(ptCoord);
		ptSet->update();
		

		this->stateY()->assign(cy);

		this->stateValueY()->setValue(ptCoord[0].y);

	}



public:
	DEF_INSTANCE_STATE(PointSet<DataType3f>, PointSet, "");

	DEF_VAR_STATE(Real, ValueY, 0, "Elapsed Time");

	DEF_ARRAY_STATE(Real, Y, DeviceType::GPU, "");

	void resetStates() override 
	{

		this->stateY()->resize(50);
		this->modifyPoints();
	}
	void updateStates() override 
	{
		this->modifyPoints();

	}

private:
	Transform3f								transformCPU;
	FArray<Transform3f, DeviceType::GPU>	transform;

	FInstance<TopologyModule>				topology;

	// bounding box of original triangle set
	NBoundingBox							bbox;
};


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
	ghost->stateForce()->resize(num);

	ghost->stateNormal()->resize(num);
	ghost->stateAttribute()->resize(num);

	ghost->statePosition()->assign(host_pos);
	ghost->stateVelocity()->assign(host_vel);
	ghost->stateForce()->assign(host_force);
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

	auto boundary = scn->addNode(std::make_shared<StaticBoundary<DataType3f>>());
	boundary->loadCube(Vec3f(-0.1f, 0.0f, -0.1f), Vec3f(0.1f, 1.0f, 0.1f), 0.005, true);
	//root->loadSDF(getAssetPath() + "bowl/bowl.sdf", false);

	auto fluid = scn->addNode(std::make_shared<ParticleSystem<DataType3f>>());
	fluid->loadParticles(Vec3f(-0.1, 0.0, -0.1), Vec3f(0.105, 0.1, 0.105), 0.005);

	auto ghost = scn->addNode(createGhostParticles());

	auto incompressibleFluid = scn->addNode(std::make_shared<GhostFluid<DataType3f>>());
	fluid->connect(incompressibleFluid->importFluidParticles());
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

		fluid->stateVelocity()->connect(calculateNorm->inVec());
		calculateNorm->outNorm()->connect(colorMapper->inScalar());

		fluid->graphicsPipeline()->pushModule(calculateNorm);
		fluid->graphicsPipeline()->pushModule(colorMapper);

		auto ptRender = std::make_shared<GLPointVisualModule>();
		ptRender->setColor(Color(1, 0, 0));
		ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);

		fluid->statePointSet()->connect(ptRender->inPointSet());
		colorMapper->outColor()->connect(ptRender->inColor());

		fluid->graphicsPipeline()->pushModule(ptRender);

		{
			auto ghostRender = std::make_shared<GLPointVisualModule>();
			ghostRender->setColor(Color(1, 0.5, 0));
			ghostRender->setColorMapMode(GLPointVisualModule::PER_OBJECT_SHADER);

			ghost->statePointSet()->connect(ghostRender->inPointSet());

			ghost->graphicsPipeline()->pushModule(ghostRender);

		}
		// A simple chart bar widget for node
		auto chart = std::make_shared<ImChart>();
		chart->varMax()->setValue(5.0f);
		calculateNorm->outNorm()->connect(chart->inArray());
		chart->varInputMode()->setCurrentKey(ImChart::InputMode::Array);
		fluid->graphicsPipeline()->pushModule(chart);
		chart->varTitle()->setValue("Fluid");
		incompressibleFluid->stateFrameNumber()->connect(chart->inFrameNumber());
		chart->varCount()->setValue(50);
	}
	
	{
		auto sinPt = scn->addNode(std::make_shared<PointFromVector>());

		//InputMode == ImChart::InputMode::Array
		//Display the position.y of each point by id;
		auto chartPt = std::make_shared<ImChart>();
		chartPt->varMax()->setValue(5.0f);
		sinPt->graphicsPipeline()->pushModule(chartPt);
		chartPt->varInputMode()->setCurrentKey(ImChart::InputMode::Array);
		sinPt->stateY()->connect(chartPt->inArray());
		chartPt->varTitle()->setValue("Position");
		sinPt->stateFrameNumber()->connect(chartPt->inFrameNumber());
		chartPt->varCount()->setValue(100); 

		//InputMode == ImChart::InputMode::Var
		//Display the position.y of Point[0] by FrameNumber;
		auto timeChart = std::make_shared<ImChart>();
		timeChart->varMax()->setValue(1.0f);
		sinPt->graphicsPipeline()->pushModule(timeChart);
		sinPt->stateValueY()->connect(timeChart->inValue());
		sinPt->stateFrameNumber()->connect(timeChart->inFrameNumber());
		timeChart->varInputMode()->setCurrentKey(ImChart::InputMode::Var);
		timeChart->varCount()->setValue(200);
		timeChart->varTitle()->setValue("Time");
	}





	return scn;
}

int main()
{


	QtApp app;

	app.setSceneGraph(createScene());

	app.initialize(1024, 768);

	//Set the distance unit for the camera, the fault unit is meter
	app.renderWindow()->getCamera()->setUnitScale(1.0);

	app.mainLoop();

	return 0;
}


