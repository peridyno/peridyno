#include "WMainApp.h"
#include "WMainWindow.h"
#include "WSimulationCanvas.h"

#include <Wt/WLinkedCssStyleSheet.h>
#include <Wt/WEnvironment.h>
#include <Wt/WHBoxLayout.h>
#include <Wt/WBootstrap5Theme.h>
#include <Wt/WBootstrap3Theme.h>

// for test data
//#include <SceneGraphFactory.h>

using namespace dyno;

WMainApp::WMainApp(const Wt::WEnvironment& env) : Wt::WApplication(env)
{
	// ace editor
	this->require("lib/ace.js");

	// use default bootstrap theme
	auto bootstrapTheme = std::make_shared<Wt::WBootstrap3Theme>();
	this->setTheme(bootstrapTheme);

	this->setTitle("PeriDyno: An AI-targeted physics simulation platform");
	//this->addMetaHeader("icon", "/logo-favicon.ico", "image/x-icon");

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

	// scrollable style
	this->styleSheet().addRule(
		".scrollable-content",
		"overflow: auto;"
		"max-height: 250px;"
		"border: 1px solid #ccc;"
		"padding: 10px;"
		"box-sizing: border-box;"
	);

	this->styleSheet().addRule(
		".scrollable-content-sample",
		"overflow: auto;"
		"max-height: auto;"
		"border: 1px solid #ccc;"
		"padding: 10px;"
		"box-sizing: border-box;"
	);

	// set layout and add main window
	auto layout = this->root()->setLayout(std::make_unique<Wt::WHBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);

	window = layout->addWidget(std::make_unique<WMainWindow>());

	window->setScene(SceneGraphFactory::instance()->createDefaultScene());

	window->createRightPanel();

	this->globalKeyWentDown().connect(window->simCanvas(), &WSimulationCanvas::onKeyWentDown);
	this->globalKeyWentUp().connect(window->simCanvas(), &WSimulationCanvas::onKeyWentUp);
	this->globalKeyWentDown().connect(window, &WMainWindow::onKeyWentDown);
}

WMainApp::~WMainApp()
{
	Wt::log("warning") << "stop WApplication";
}