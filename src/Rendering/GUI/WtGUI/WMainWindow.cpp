#include "WMainWindow.h"

#define WIDTH_SCALE 0.4
#define HEIGHT_SCALE 0.75

WMainWindow::WMainWindow()
	: WContainerWidget(), bRunFlag(false)
{
	// disable page margin...
	setMargin(0);
	auto layout = this->setLayout(std::make_unique<Wt::WBorderLayout>());
	layout->setContentsMargins(0, 0, 0, 0);

	pythonWidget = new WPythonWidget();
	viewportHeight = Wt::WApplication::instance()->environment().screenHeight();
	viewportWidth = Wt::WApplication::instance()->environment().screenWidth();

	//create a navigation bar
	auto naviBar = layout->addWidget(std::make_unique<Wt::WNavigationBar>(), Wt::LayoutPosition::North);
	naviBar->addStyleClass("main-nav");
	naviBar->setResponsive(true);
	naviBar->setTitle("PeriDyno", "https://github.com/peridyno/peridyno");
	naviBar->setMargin(0);

	// create center
	auto centerContainer = layout->addWidget(std::make_unique<Wt::WContainerWidget>(), Wt::LayoutPosition::Center);
	centerContainer->setMargin(0);

	auto centerVbox = centerContainer->setLayout(std::make_unique<Wt::WVBoxLayout>());
	centerVbox->setContentsMargins(0, 0, 0, 0);

	mSceneCanvas = centerVbox->addWidget(std::make_unique<WSimulationCanvas>());
	auto controlContainer = centerVbox->addWidget(std::make_unique<Wt::WContainerWidget>());
	controlContainer->setMargin(0);
	initSimulationControl(controlContainer);
	// central canvas
	//mSceneCanvas = layout->addWidget(std::make_unique<WSimulationCanvas>(), Wt::LayoutPosition::Center);

	// scene info panel
	rightWidget = layout->addWidget(std::make_unique<Wt::WContainerWidget>(), Wt::LayoutPosition::East);
	//rightWidget->setWidth(viewportWidth * WIDTH_SCALE);

	// bottom
	//bottomWidget = layout->addWidget(std::make_unique<Wt::WContainerWidget>(), Wt::LayoutPosition::South);
	//initSimulationControl(bottomWidget);

	// menu
	auto widget1 = layout->addWidget(std::make_unique<Wt::WStackedWidget>(), Wt::LayoutPosition::West);

	// create data model
	mNodeDataModel = std::make_shared<WNodeDataModel>();
	mModuleDataModel = std::make_shared<WModuleDataModel>();
	mParameterDataNode = std::make_shared<WParameterDataNode>();
	mParameterDataNode->changeValue().connect(this, &WMainWindow::updateCanvas);


	//std::cout << rightWidget->height() << std::endl;

}

WMainWindow::~WMainWindow()
{
	Wt::log("warning") << "stop WMainWindows";
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

void WMainWindow::createRightPanel()
{
	initRightPanel(rightWidget);
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


void WMainWindow::initRightPanel(Wt::WContainerWidget* parent)
{
	// vertical layout
	auto layout = parent->setLayout(std::make_unique<Wt::WVBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	parent->setMargin(0);

	auto widget0 = layout->addWidget(std::make_unique<Wt::WContainerWidget>());
	widget0->resize("100%", "100%");
	widget0->setMargin(0);
	tab = widget0->addNew<Wt::WTabWidget>();
	tab->resize("100%", "100%");
	tab->addTab(initNodeGraphics(), "NodeGraphics", Wt::ContentLoading::Eager);
	tab->addTab(initPython(), "Python", Wt::ContentLoading::Eager);
	tab->addTab(initSample(), "Sample", Wt::ContentLoading::Lazy);
	tab->addTab(initSave(), "Save", Wt::ContentLoading::Lazy);
	tab->addTab(initLog(), "Log", Wt::ContentLoading::Lazy);
}

void WMainWindow::initSimulationControl(Wt::WContainerWidget* parent)
{
	// vertical layout
	auto layout = parent->setLayout(std::make_unique<Wt::WVBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	parent->setMargin(0);

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

void WMainWindow::initAddNodePanel(Wt::WPanel* panel)
{
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

	auto nodeMap = dyno::Object::getClassMap();
	for (auto it = nodeMap->begin(); it != nodeMap->end(); ++it)
	{
		auto node_obj = dyno::Object::createObject(it->second->m_className);
		std::shared_ptr<dyno::Node> new_node(dynamic_cast<dyno::Node*>(node_obj));
		if (new_node == nullptr)
		{
			continue;
		}
		else
		{
			sp->addSuggestion(it->second->m_className);
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
							auto new_node = mScene->addNode(action->action()());
							new_node->setBlockCoord(Initial_x, Initial_y);
							Initial_x += 20;
							Initial_y += 20;
							name->setText("");
							mFlowWidget->updateForAddNode();
							mNodeDataModel->setScene(mScene);
							flag = false;
						}
					}
				}
			}
		}

		if (flag)
		{
			auto node_obj = dyno::Object::createObject(name->text().toUTF8());
			std::shared_ptr<dyno::Node> new_node(dynamic_cast<dyno::Node*>(node_obj));
			if (new_node != nullptr)
			{
				mScene->addNode(new_node);
				new_node->setBlockCoord(Initial_x, Initial_y);
				Initial_x += 10;
				Initial_y += 10;
				std::cout << Initial_x << std::endl;
				std::cout << "!!!!!!!!!!!" << std::endl;
				mFlowWidget->updateForAddNode();
				mNodeDataModel->setScene(mScene);
				name->setText("");
			}

		}
		});

	auto reorderNodeButton = layout3->addWidget(std::make_unique<Wt::WPushButton>("Reorder"));

	reorderNodeButton->clicked().connect([=] {
		mFlowWidget->reorderNode();
		});
}

std::unique_ptr<Wt::WWidget> WMainWindow::initNodeGraphics()
{
	auto rootWidget = std::make_unique<Wt::WContainerWidget>();
	auto layout = rootWidget->setLayout(std::make_unique<Wt::WVBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	rootWidget->setMargin(0);

	// add node
	auto panel = layout->addWidget(std::make_unique<Wt::WPanel>());
	panel->setTitle("Add Node");
	panel->setCollapsible(false);
	initAddNodePanel(panel);

	auto panel0 = layout->addWidget(std::make_unique<Wt::WPanel>());
	panel0->setTitleBar(false);
	panel0->setCollapsible(false);
	panel0->setMargin(0);

	if (mScene)
	{
		auto painteContainer = panel0->setCentralWidget(std::make_unique<Wt::WContainerWidget>());
		painteContainer->setMargin(0);
		//painteContainer->resize("100%", "100%");
		mFlowWidget = painteContainer->addWidget(std::make_unique<WtFlowWidget>(mScene, this));
		mFlowWidget->resize(viewportWidth * WIDTH_SCALE, viewportHeight * 0.4);
		//mFlowWidget = panel0->setCentralWidget(std::make_unique<WtFlowWidget>(mScene, this));
	}

	// module list
	auto panel2 = layout->addWidget(std::make_unique<Wt::WPanel>());
	panel2->setTitle("Module List");
	panel2->setCollapsible(true);
	panel2->setStyleClass("scrollable-content");
	auto tableView = panel2->setCentralWidget(std::make_unique<Wt::WTableView>());

	tableView->setSortingEnabled(false);
	tableView->setSelectionMode(Wt::SelectionMode::Single);
	tableView->setEditTriggers(Wt::EditTrigger::None);
	tableView->setModel(mModuleDataModel);

	// Parameter list
	auto panel3 = layout->addWidget(std::make_unique<Wt::WPanel>());
	panel3->setTitle("Control Variable");
	panel3->setCollapsible(true);
	panel3->setStyleClass("scrollable-content");

	//action for selection change
	mFlowWidget->selectNodeSignal().connect([=](int selectNum)
		{
			if (selectNum < 0)
			{
				std::cout << "selectNum:" << selectNum << std::endl;
			}
			else
			{

				for (auto it = mScene->begin(); it != mScene->end(); it++)
				{
					auto m = it.get();
					if (m->objectId() == selectNum)
					{
						mModuleDataModel->setNode(m);
						mParameterDataNode->setNode(m);
						mParameterDataNode->createParameterPanel(panel3);
					}
				}
			}
		});

	tableView->clicked().connect([=](const Wt::WModelIndex& idx, const Wt::WMouseEvent& evt)
		{
			auto module = mModuleDataModel->getModule(idx);
			mParameterDataNode->setModule(module);
			mParameterDataNode->createParameterPanelModule(panel3);
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

std::unique_ptr<Wt::WWidget> WMainWindow::initPython()
{
	pythonWidget->resize(viewportWidth * WIDTH_SCALE, viewportHeight * HEIGHT_SCALE);

	pythonWidget->updateSceneGraph().connect([=](std::shared_ptr<dyno::SceneGraph> scene) {
		if (scene)
		{
			std::cout << "delete" << std::endl;
			setScene(scene);
			//initLeftPanel(widget0);
			//initNodeGraphics();
			tab->removeTab(tab->widget(0));
			tab->insertTab(0, initNodeGraphics(), "NodeGraphics", Wt::ContentLoading::Lazy);
		}
		});

	return std::unique_ptr<WPythonWidget>(pythonWidget);
}

std::unique_ptr<Wt::WWidget> WMainWindow::initSample()
{
	int maxColumns = viewportWidth * WIDTH_SCALE / 200;

	auto sampleWidget = new WSampleWidget(maxColumns);
	sampleWidget->setStyleClass("scrollable-content-sample");
	sampleWidget->setHeight(viewportHeight * HEIGHT_SCALE);
	sampleWidget->setWidth(viewportWidth * WIDTH_SCALE);

	sampleWidget->clicked().connect([=](Sample* sample)
		{
			if (sample != NULL)
			{
				//pythonItem->select();
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

	return std::unique_ptr<WSampleWidget>(sampleWidget);
}

std::unique_ptr<Wt::WWidget> WMainWindow::initSave()
{
	auto rootWidget = std::make_unique<Wt::WContainerWidget>();

	rootWidget->resize(viewportWidth * WIDTH_SCALE, viewportHeight * HEIGHT_SCALE);

	rootWidget->setMargin(0);
	rootWidget->setContentAlignment(Wt::AlignmentFlag::Middle);

	auto saveWidget = rootWidget->addWidget(std::make_unique<WSaveWidget>(this, viewportWidth * WIDTH_SCALE / 2));

	saveWidget->resize("100%", "100%");

	Wt::WApplication::instance()->styleSheet().addRule(
		".save-middle",
		"justify-items: anchor-center;"
	);

	saveWidget->setStyleClass("save-middle");

	return rootWidget;
}

std::unique_ptr<Wt::WWidget> WMainWindow::initLog()
{
	auto logWidget = new WLogWidget(this);
	logWidget->resize(viewportWidth * WIDTH_SCALE, viewportHeight * HEIGHT_SCALE);

	auto logMessage = new WLogMessage();

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

	return std::unique_ptr<WLogWidget>(logWidget);
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