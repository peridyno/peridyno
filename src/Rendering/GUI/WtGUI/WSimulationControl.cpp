#include "WSimulationControl.h"
#include <Wt/WApplication.h>
#include <Wt/WHBoxLayout.h>
#include <Wt/WPanel.h>
#include <Wt/WPushButton.h>
#include <Wt/WVBoxLayout.h>

WSimulationControl::WSimulationControl()
{
	// vertical layout
	auto layout = this->setLayout(std::make_unique<Wt::WVBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	this->setMargin(0);

	// simulation control
	auto panel3 = layout->addWidget(std::make_unique<Wt::WPanel>());
	panel3->setTitle("Simulation Control");
	panel3->setCollapsible(false);
	//panel3->setHeight(50);
	auto widget2 = panel3->setCentralWidget(std::make_unique<Wt::WContainerWidget>());
	auto layout2 = widget2->setLayout(std::make_unique<Wt::WHBoxLayout>());
	//widget2->setHeight(5);
	layout2->setContentsMargins(0, 0, 0, 0);
	auto startButton = layout2->addWidget(std::make_unique<Wt::WPushButton>("Start"));
	auto stopButton = layout2->addWidget(std::make_unique<Wt::WPushButton>("Stop"));
	auto stepButton = layout2->addWidget(std::make_unique<Wt::WPushButton>("Step"));
	auto resetButton = layout2->addWidget(std::make_unique<Wt::WPushButton>("Reset"));

	startButton->setId("startButton");
	stopButton->setId("stopButton");
	stepButton->setId("stepButton");
	resetButton->setId("resetButton");

	// actions
	startButton->clicked().connect(this, &WSimulationControl::start);
	stopButton->clicked().connect(this, &WSimulationControl::stop);
	stepButton->clicked().connect(this, &WSimulationControl::step);
	resetButton->clicked().connect(this, &WSimulationControl::reset);

	stopButton->clicked().connect([=] {
		stopButton->doJavaScript("var stopButton = document.getElementById('stopButton');"
			"stopButton.blur();");
		});
	stepButton->clicked().connect([=] {
		stepButton->doJavaScript("var stepButton = document.getElementById('stepButton');"
			"stepButton.blur();");
		});
	resetButton->clicked().connect([=] {
		resetButton->doJavaScript("var resetButton = document.getElementById('resetButton');"
			"resetButton.blur();");
		});
}

WSimulationControl::~WSimulationControl() {}

void WSimulationControl::start()
{
	//startButton->doJavaScript("var startButton = document.getElementById('startButton');"
	//	"startButton.blur();");
	if (mScene != nullptr && mSceneCanvas != nullptr)
	{
		mSceneCanvas->setFocus();
		if (mReset)
		{
			mScene->reset();
			mReset = false;
		}
		this->bRunFlag = true;

		Wt::WApplication* app = Wt::WApplication::instance();

		while (this->bRunFlag)
		{
			mScene->takeOneFrame();
			mSceneCanvas->update();
			Wt::log("info") << "Step!!!";
			Wt::log("info") << mScene->getFrameNumber();
			app->processEvents();
		}
	}
}

void WSimulationControl::stop()
{
	if (mSceneCanvas != nullptr)
	{
		mSceneCanvas->setFocus(true);
		this->bRunFlag = false;
	}
}

void WSimulationControl::step()
{
	if (mScene != nullptr && mSceneCanvas != nullptr)
	{
		stop();
		mScene->takeOneFrame();
		mSceneCanvas->update();
	}

	Wt::log("info") << "Step!!!";
	Wt::log("info") << mScene->getFrameNumber();
}

void WSimulationControl::reset()
{
	if (mScene != nullptr && mSceneCanvas != nullptr)
	{
		this->bRunFlag = false;

		mScene->setFrameNumber(0);
		mScene->reset();
		mSceneCanvas->update();

		mSceneCanvas->update();

		mReset = true;
	}

	Wt::log("info") << mScene->getFrameNumber();
}