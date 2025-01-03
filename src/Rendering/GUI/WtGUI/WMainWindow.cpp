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
#include <Wt/WTabWidget.h>
#include <Wt/WTextArea.h>

#include <fstream>

#include <SceneGraph.h>
#include <SceneGraphFactory.h>

#include <filesystem>

//#include "Dynamics/Cuda/ParticleSystem/initializeParticleSystem.h"

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
	//mSceneCanvas->setScene(mScene);

	// scene info panel
	widget0 = layout->addWidget(std::make_unique<Wt::WContainerWidget>(), Wt::LayoutPosition::East);
	widget0->setWidth(900);

	// create data model
	mNodeDataModel = std::make_shared<WNodeDataModel>();
	mModuleDataModel = std::make_shared<WModuleDataModel>();
	mParameterDataNode = std::make_shared<WParameterDataNode>();

	mParameterDataNode->changeValue().connect(this, &WMainWindow::updateCanvas);

	// menu
	auto widget1 = layout->addWidget(std::make_unique<Wt::WStackedWidget>(), Wt::LayoutPosition::West);
	auto menu = naviBar->addMenu(std::make_unique<Wt::WMenu>(widget1), Wt::AlignmentFlag::Right);
	initMenu(menu);

}

WMainWindow::~WMainWindow()
{
	Wt::log("warning") << "stop WMainWindows";
}

void WMainWindow::createLeftPanel()
{
	initLeftPanel(widget0);
}

void WMainWindow::initMenu(Wt::WMenu* menu)
{
	menu->setMargin(5, Wt::Side::Right);

	auto sampleWidget = new WSampleWidget();
	auto pythonWidget = new WPythonWidget();
	auto saveWidget = new WSaveWidget(this);
	auto logWidget = new WLogWidget(this);
	auto logMessage = new WLogMessage();

	//auto paramsWidget = new WRenderParamsWidget(&mSceneCanvas->getRenderParams());
	//menu->addItem("Settings", std::unique_ptr<WRenderParamsWidget>(paramsWidget));

	/*paramsWidget->valueChanged().connect([=]() {
		mSceneCanvas->update();
		});*/

	menu->addItem("Samples", std::unique_ptr<WSampleWidget>(sampleWidget));

	auto pythonItem = menu->addItem("Python", std::unique_ptr<WPythonWidget>(pythonWidget));

	auto saveItem = menu->addItem("Save", std::unique_ptr<WSaveWidget>(saveWidget));

	auto lgoItem = menu->addItem("Log", std::unique_ptr<WLogWidget>(logWidget));

	pythonWidget->updateSceneGraph().connect([=](std::shared_ptr<dyno::SceneGraph> scene) {
		if (scene)
		{
			std::cout << "delete" << std::endl;
			setScene(scene);
			initLeftPanel(widget0);
		}

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


	logMessage->updateText().connect([=](std::string message)
		{
			std::ostringstream oss;
			oss << mScene;
			std::string filename = oss.str() + ".txt";
			std::ofstream fileStream(filename, std::ios::out | std::ios::trunc);
			if (fileStream.is_open()) {
				fileStream << message;
				fileStream.close();
			}
			else {
				std::cerr << "Unable to open file for writing." << std::endl;
			}

		});

	auto hide = menu->addItem(">>", 0);
	hide->select();
	hide->clicked().connect([=]() {
		menu->contentsStack()->setCurrentWidget(0);
		});
}

std::unique_ptr<Wt::WWidget> WMainWindow::initNodeGraphics()
{
	auto panel0 = std::make_unique<Wt::WPanel>();
	panel0->setTitleBar(false);
	panel0->setCollapsible(false);
	panel0->setMargin(0);
	//panel0->setHeight(900);

	if (mScene)
	{
		//setScene(scn);
		mFlowWidget = panel0->setCentralWidget(std::make_unique<WtFlowWidget>(mScene, this));
	}

	return panel0;
}

std::unique_ptr<Wt::WWidget> WMainWindow::initNodeTree()
{
	auto rootWidget = std::make_unique<Wt::WContainerWidget>();
	//rootWidget->setHeight(900);
	// vertical layout
	auto layout = rootWidget->setLayout(std::make_unique<Wt::WVBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	rootWidget->setMargin(0);



	// node tree
	auto panel0 = layout->addWidget(std::make_unique<Wt::WPanel>());
	panel0->setTitle("Node Tree");
	panel0->setCollapsible(true);
	panel0->setMargin(0);
	//panel0->setCentralWidget(std::make_unique<WtFlowWidget>());
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
	treeView->setSortingEnabled(false);

	// module list
	auto panel1 = layout->addWidget(std::make_unique<Wt::WPanel>());
	panel1->setTitle("Module List");
	panel1->setCollapsible(true);
	//panel1->setStyleClass("scrollable-content");
	auto tableView = panel1->setCentralWidget(std::make_unique<Wt::WTableView>());

	tableView->setSortingEnabled(false);
	tableView->setSelectionMode(Wt::SelectionMode::Single);
	tableView->setEditTriggers(Wt::EditTrigger::None);
	tableView->setModel(mModuleDataModel);

	// Parameter list
	auto panel2 = layout->addWidget(std::make_unique<Wt::WPanel>());
	panel2->setTitle("Control Variable");
	panel2->setCollapsible(true);
	//panel2->setStyleClass("scrollable-content");

	//action for selection change
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
	return rootWidget;
}

void WMainWindow::initLeftPanel(Wt::WContainerWidget* parent)
{
	// vertical layout
	auto layout = parent->setLayout(std::make_unique<Wt::WVBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	parent->setMargin(0);

	auto widget0 = layout->addWidget(std::make_unique<Wt::WContainerWidget>(), 1);
	Wt::WTabWidget* tab = widget0->addNew<Wt::WTabWidget>();
	tab->setHeight(900);
	tab->setWidth(900);
	tab->addTab(initNodeGraphics(), "NodeGraphics", Wt::ContentLoading::Lazy);
	tab->addTab(initNodeTree(), "NodeTree", Wt::ContentLoading::Eager);

	// add node
	auto panel = layout->addWidget(std::make_unique<Wt::WPanel>());
	panel->setTitle("Add Node");
	panel->setCollapsible(false);

	auto widget3 = panel->setCentralWidget(std::make_unique<Wt::WContainerWidget>());

	auto layout3 = widget3->setLayout(std::make_unique<Wt::WHBoxLayout>());
	layout3->setContentsMargins(0, 0, 0, 0);

	Wt::WSuggestionPopup::Options nodeOptions;
	nodeOptions.highlightBeginTag = "<span class=\"highlight\">";
	nodeOptions.highlightEndTag = "</span>";

	Wt::WSuggestionPopup* sp = layout3->addChild(std::make_unique<Wt::WSuggestionPopup>(
		Wt::WSuggestionPopup::generateMatcherJS(nodeOptions),
		Wt::WSuggestionPopup::generateReplacerJS(nodeOptions)
	));


	auto& pages = dyno::NodeFactory::instance()->nodePages();
	for (auto iPage = pages.begin(); iPage != pages.end(); iPage++)
	{
		auto& groups = iPage->second->groups();
		{
			for (auto iGroup = groups.begin(); iGroup != groups.end(); iGroup++)
			{
				auto& actions = iGroup->second->actions();
				for (auto action : actions)
				{
					sp->addSuggestion(action->caption());
				}
			}
		}
	}

	auto name = layout3->addWidget(std::make_unique<Wt::WLineEdit>());
	name->setPlaceholderText("node name");

	sp->forEdit(name);

	auto addNodeButton = layout3->addWidget(std::make_unique<Wt::WPushButton>("Add"));

	addNodeButton->clicked().connect([=] {
		bool flag = true;

		for (auto iPage = pages.begin(); iPage != pages.end(); iPage++)
		{
			auto& groups = iPage->second->groups();
			{
				for (auto iGroup = groups.begin(); iGroup != groups.end(); iGroup++)
				{
					auto& actions = iGroup->second->actions();
					for (auto action : actions)
					{
						if (action->caption() == name->text().toUTF8())
						{
							mScene->addNode(action->action()());
							name->setText("");
							mFlowWidget->updateForAddNode();
							mNodeDataModel->setScene(mScene);
							flag = false;
							std::cout << "add" << std::endl;
						}
					}
				}
			}
		}

		if (flag)
		{
			auto node_obj = dyno::Object::createObject(name->text().toUTF8());
			std::shared_ptr<dyno::Node> new_node(dynamic_cast<dyno::Node*>(node_obj));
			mScene->addNode(new_node);
			mFlowWidget->updateForAddNode();
			mNodeDataModel->setScene(mScene);
			name->setText("");
		}

		});

	auto reorderNodeButton = layout3->addWidget(std::make_unique<Wt::WPushButton>("Reorder"));

	reorderNodeButton->clicked().connect([=] {
		mFlowWidget->reorderNode();
		});

	// simulation control
	auto panel3 = layout->addWidget(std::make_unique<Wt::WPanel>());
	panel3->setTitle("Simulation Control");
	panel3->setCollapsible(false);
	//panel3->setHeight(50);
	auto widget2 = panel3->setCentralWidget(std::make_unique<Wt::WContainerWidget>());
	auto layout2 = widget2->setLayout(std::make_unique<Wt::WHBoxLayout>());
	//widget2->setHeight(5);
	layout2->setContentsMargins(0, 0, 0, 0);
	startButton = layout2->addWidget(std::make_unique<Wt::WPushButton>("Start"));
	auto stopButton = layout2->addWidget(std::make_unique<Wt::WPushButton>("Stop"));
	auto stepButton = layout2->addWidget(std::make_unique<Wt::WPushButton>("Step"));
	auto resetButton = layout2->addWidget(std::make_unique<Wt::WPushButton>("Reset"));

	startButton->setId("startButton");
	stopButton->setId("stopButton");
	stepButton->setId("stepButton");
	resetButton->setId("resetButton");

	// actions
	startButton->clicked().connect(this, &WMainWindow::start);
	stopButton->clicked().connect(this, &WMainWindow::stop);
	stepButton->clicked().connect(this, &WMainWindow::step);
	resetButton->clicked().connect(this, &WMainWindow::reset);

	//startButton->clicked().connect([=] {
	//	startButton->doJavaScript("var startButton = document.getElementById('startButton');"
	//		"startButton.blur();");
	//	});

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

void WMainWindow::start()
{
	startButton->doJavaScript("var startButton = document.getElementById('startButton');"
		"startButton.blur();");
	if (mScene)
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

void WMainWindow::stop()
{
	mSceneCanvas->setFocus(true);
	this->bRunFlag = false;
}

void WMainWindow::step()
{
	if (mScene)
	{
		stop();
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

void WMainWindow::onKeyWentDown(const Wt::WKeyEvent& event)
{
	if (event.key() == Wt::Key::Delete || event.key() == Wt::Key::Backspace)
	{
		mFlowWidget->onKeyWentDown();
	}
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

std::shared_ptr<dyno::SceneGraph> WMainWindow::getScene()
{
	return mScene;
}

