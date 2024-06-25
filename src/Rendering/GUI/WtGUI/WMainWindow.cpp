#include "WMainWindow.h"
#include "WSceneDataModel.h"
#include "WSimulationCanvas.h"
#include "WSampleWidget.h"
#include "WRenderParamsWidget.h"
#include "WPythonWidget.h"
#include "WParameterDataNode.h"

#include <Wt/WVBoxLayout.h>
#include <Wt/WHBoxLayout.h>
#include <Wt/WBorderLayout.h>
#include <Wt/WPushButton.h>
#include <Wt/WPanel.h>
#include <Wt/WApplication.h>
#include <Wt/WMenu.h>
#include <Wt/WPopupMenu.h>
#include <Wt/WNavigationBar.h>
#include <Wt/WTreeView.h>
#include <Wt/WTableView.h>
#include <Wt/WStackedWidget.h>
#include <Wt/WText.h>
#include <Wt/WTable.h>
#include <Wt/WColorPicker.h>
#include <Wt/WLogger.h>

#include <fstream>

#include <SceneGraph.h>
#include <SceneGraphFactory.h>

#include <filesystem>

WMainWindow::WMainWindow()
	: WContainerWidget(), bRunFlag(false)
{
	// disable page margin...
	setMargin(0);

	auto layout = this->setLayout(std::make_unique<Wt::WBorderLayout>());
	layout->setContentsMargins(0, 0, 0, 0);

	//create a navigation bar
	auto naviBar = layout->addWidget(std::make_unique<Wt::WNavigationBar>(), Wt::LayoutPosition::North);
	naviBar->addStyleClass("main-nav");
	naviBar->setTitle("PeriDyno", "https://github.com/peridyno/peridyno");
	naviBar->setMargin(0);

	// central canvas
	mSceneCanvas = layout->addWidget(std::make_unique<WSimulationCanvas>(), Wt::LayoutPosition::Center);

	// scene info panel
	auto widget0 = layout->addWidget(std::make_unique<Wt::WContainerWidget>(), Wt::LayoutPosition::West);
	widget0->setWidth(400);
	initLeftPanel(widget0);

	// menu
	auto widget1 = layout->addWidget(std::make_unique<Wt::WStackedWidget>(), Wt::LayoutPosition::East);
	auto menu = naviBar->addMenu(std::make_unique<Wt::WMenu>(widget1), Wt::AlignmentFlag::Right);
	initMenu(menu);
}

WMainWindow::~WMainWindow()
{
	Wt::log("warning") << "stop WMainWindows";
}

void WMainWindow::initMenu(Wt::WMenu* menu)
{
	menu->setMargin(5, Wt::Side::Right);

	auto sampleWidget = new WSampleWidget();
	auto paramsWidget = new WRenderParamsWidget(&mSceneCanvas->getRenderParams());
	auto pythonWidget = new WPythonWidget;

	menu->addItem("Samples", std::unique_ptr<WSampleWidget>(sampleWidget));
	menu->addItem("Settings", std::unique_ptr<WRenderParamsWidget>(paramsWidget));
	auto pythonItem = menu->addItem("Python", std::unique_ptr<WPythonWidget>(pythonWidget));

	paramsWidget->valueChanged().connect([=]() {
		mSceneCanvas->update();
		});

	pythonWidget->updateSceneGraph().connect([=](std::shared_ptr<dyno::SceneGraph> scene) {
		if (scene) setScene(scene);
		});

	sampleWidget->clicked().connect([=](Sample* sample)
		{
			if (sample != NULL)
			{
				pythonItem->select();

				std::string path = sample->source();
				std::ifstream ifs(path);
				if (ifs.is_open())
				{
					std::string content((std::istreambuf_iterator<char>(ifs)),
						(std::istreambuf_iterator<char>()));
					pythonWidget->setText(content);
					pythonWidget->execute(content);
					//menu->contentsStack()->setCurrentWidget(0);
				}
				else
				{
					std::string content = "Error: Not Find The Python File";
					pythonWidget->setText(content);
				}
			}
		});

	auto hide = menu->addItem(">>", 0);
	hide->select();
	hide->clicked().connect([=]() {
		menu->contentsStack()->setCurrentWidget(0);
		});
}

void WMainWindow::initLeftPanel(Wt::WContainerWidget* parent)
{
	// create data model
	mNodeDataModel = std::make_shared<WNodeDataModel>();
	mModuleDataModel = std::make_shared<WModuleDataModel>();
	mParameterDataNode = std::make_shared<WParameterDataNode>();

	mParameterDataNode->changeValue().connect(this, &WMainWindow::updateCanvas);

	// vertical layout

	auto layout = parent->setLayout(std::make_unique<Wt::WVBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	parent->setMargin(0);

	// node tree
	auto panel0 = layout->addWidget(std::make_unique<Wt::WPanel>(), 2);
	panel0->setTitle("Node Tree");
	panel0->setCollapsible(true);
	panel0->setMargin(0);
	//panel0->setStyleClass("scrollable-content");

	auto treeView = panel0->setCentralWidget(std::make_unique<Wt::WTreeView>());
	treeView->setMargin(0);
	treeView->setSortingEnabled(false);
	treeView->setSelectionMode(Wt::SelectionMode::Single);
	treeView->setEditTriggers(Wt::EditTrigger::None);
	treeView->setColumnResizeEnabled(true);
	treeView->setModel(mNodeDataModel);
	treeView->setColumnWidth(0, 100);
	treeView->setColumnWidth(1, 280);

	// module list
	auto panel1 = layout->addWidget(std::make_unique<Wt::WPanel>(), 2);
	panel1->setTitle("Module List");
	panel1->setCollapsible(true);
	panel1->setStyleClass("scrollable-content");
	auto tableView = panel1->setCentralWidget(std::make_unique<Wt::WTableView>());
	treeView->setSortingEnabled(false);
	tableView->setSortingEnabled(false);
	tableView->setSelectionMode(Wt::SelectionMode::Single);
	tableView->setEditTriggers(Wt::EditTrigger::None);
	tableView->setModel(mModuleDataModel);

	// Parameter list
	auto panel2 = layout->addWidget(std::make_unique<Wt::WPanel>(), 6);
	panel2->setTitle("Control Variable");
	panel2->setCollapsible(true);
	panel2->setStyleClass("scrollable-content");

	// action for selection change
	treeView->clicked().connect([=](const Wt::WModelIndex& idx, const Wt::WMouseEvent& evt)
		{
			auto node = mNodeDataModel->getNode(idx);
			mModuleDataModel->setNode(node);
			mParameterDataNode->setNode(node);
			mParameterDataNode->createParameterPanel(panel2);
		});

	tableView->clicked().connect([=](const Wt::WModelIndex& idx, const Wt::WMouseEvent& evt)
		{
			auto module = mModuleDataModel->getModule(idx);
			mParameterDataNode->setModule(module);
			mParameterDataNode->createParameterPanelModule(panel2);
		});

	tableView->doubleClicked().connect([=](const Wt::WModelIndex& idx, const Wt::WMouseEvent& evt)
		{
			auto mod = mModuleDataModel->getModule(idx);
			if (mod->getModuleType() == "VisualModule")
			{
				Wt::log("info") << mod->getName();
			}
		});

	// simulation control
	auto panel3 = layout->addWidget(std::make_unique<Wt::WPanel>(), 1);
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

	// actions
	stepButton->clicked().connect(this, &WMainWindow::step);
	startButton->clicked().connect(this, &WMainWindow::start);
	stopButton->clicked().connect(this, &WMainWindow::stop);
	resetButton->clicked().connect(this, &WMainWindow::reset);
}

void WMainWindow::start()
{
	if (mScene)
	{
		if (mReset)
		{
			mScene->reset();
			mReset = false;
		}
		this->bRunFlag = true;
		Wt::WApplication* app = Wt::WApplication::instance();
		while (this->bRunFlag)
		{
			step();
			app->processEvents();
		}
	}
}

void WMainWindow::stop()
{
	this->bRunFlag = false;
}

void WMainWindow::step()
{
	if (mScene)
	{
		mScene->takeOneFrame();
		mSceneCanvas->update();
	}

	Wt::log("info") << "Step!!!";
	Wt::log("info") << mScene->getFrameNumber();
}

void WMainWindow::reset()
{
	if (mScene)
	{
		this->bRunFlag = false;

		mScene->setFrameNumber(0);
		mScene->reset();
		mSceneCanvas->update();

		mReset = true;
	}

	Wt::log("info") << mScene->getFrameNumber();
}

void WMainWindow::updateCanvas()
{
	if (mScene)
	{
		mSceneCanvas->update();
	}
	Wt::log("info") << "updateCanvas!!!";
	Wt::log("info") << mScene->getFrameNumber();
}

void WMainWindow::setScene(std::shared_ptr<dyno::SceneGraph> scene)
{
	// try to stop the simulation
	stop();

	// setup scene graph
	mScene = scene;
	mSceneCanvas->setScene(mScene);
	mNodeDataModel->setScene(mScene);
	mModuleDataModel->setNode(NULL);
}