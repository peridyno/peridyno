#include "WMainApp.h"
#include "WMainWindow.h"

#include <Wt/WLinkedCssStyleSheet.h>
#include <Wt/WEnvironment.h>
#include <Wt/WHBoxLayout.h>
#include <Wt/WBootstrapTheme.h>

// for test data
#include <ParticleSystem/ParticleFluid.h>
#include <ParticleSystem/GhostParticles.h>
#include <ParticleSystem/StaticBoundary.h>
#include <ParticleSystem/GhostFluid.h>
#include <Module/CalculateNorm.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>

using namespace dyno;

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

WMainApp::WMainApp(const Wt::WEnvironment& env) : Wt::WApplication(env)
{
	// ace editor
	this->require("lib/ace.js");

	// use default bootstrap theme
	auto theme = std::make_shared<Wt::WBootstrapTheme>();
	theme->setVersion(Wt::BootstrapVersion::v3);
	theme->setResponsive(true);
	this->setTheme(theme);

	this->setTitle("PeriDyno: An AI-targeted physics simulation platform");


	// style sheet for the canvas
	this->styleSheet().addRule(
		".remote-framebuffer",
		// flip
		"transform: scaleY(-1) !important;"
		// disable drag...
		"-webkit-user-drag: none !important;"
		"-khtml-user-drag: none !important;"
		"-moz-user-drag: none !important;"
		"-o-user-drag: none !important;"
		"user-drag: none !important;"
		// hack for brightness
		"filter: brightness(2);"
	);

	// override internal padding for panel...
	this->styleSheet().addRule(
		".panel-body",
		"padding: 0!important;"
	);

	this->styleSheet().addRule(
		".sample-item",
		"border-radius: 5px;"
	);

	// add logo to navbar
	this->styleSheet().addRule(
		".navbar-header",
		"background-image: url(\"logo.png\");"
		"background-repeat: no-repeat;"
		"background-size: 36px 36px;"
		"background-position: 12px 6px;"
		"padding-left: 36px;"
	);

	// color picker button style
	this->styleSheet().addRule(
		".color-picker",
		"border: 0!important;"
		"padding: 0!important;"
	);

	// set layout and add main window
	auto layout = this->root()->setLayout(std::make_unique<Wt::WHBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);

	window = layout->addWidget(std::make_unique<WMainWindow>());

	// create an empty scene
	scene = std::make_shared<dyno::SceneGraph>();

	// load test data...
	if (true) {
		scene = createScene();
	}

	window->setScene(scene);

}

WMainApp::~WMainApp()
{
	Wt::log("warning") << "stop WApplication";
}
