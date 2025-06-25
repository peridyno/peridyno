#include "WMainWindow.h"

#include "NodeEditor/WtFlowWidget.h"
#include "NodeFactory.h"
#include "WLogWidget.h"

#include "WParameterDataNode.h"
#include "WSampleWidget.h"
#include "WSaveWidget.h"
#include "WSceneDataModel.h"
#include "WSimulationCanvas.h"

#include <fstream>
#include <SceneGraph.h>

#include <Wt/WApplication.h>
#include <Wt/WEnvironment.h>
#include <Wt/WHBoxLayout.h>
#include <Wt/WLineEdit.h>
#include <Wt/WLogger.h>
#include <Wt/WNavigationBar.h>
#include <Wt/WPushButton.h>
#include <Wt/WSuggestionPopup.h>
#include <Wt/WTableView.h>
#include <Wt/WVBoxLayout.h>

#define WIDTH_SCALE 0.4
#define HEIGHT_SCALE 0.75

WMainWindow::WMainWindow() : WContainerWidget()
{
	// disable page margin...
	setMargin(0);
	auto layout = this->setLayout(std::make_unique<Wt::WBorderLayout>());
	layout->setContentsMargins(0, 0, 0, 0);

	viewportHeight = Wt::WApplication::instance()->environment().screenHeight();
	viewportWidth = Wt::WApplication::instance()->environment().screenWidth();

	// init
	initNavigationBar(layout);
	initCenterContainer(layout);

	// right panel
	rightWidget = layout->addWidget(std::make_unique<Wt::WContainerWidget>(), Wt::LayoutPosition::East);

	// create data model
	mParameterDataNode = std::make_shared<WParameterDataNode>();
	mPromptPanel = std::make_shared<WPromptPanel>();

	mParameterDataNode->changeValue().connect(this, &WMainWindow::updateCanvas);
	mParameterDataNode->changeValue().connect(this, &WMainWindow::updateNodeGraphics);

	//mParameterDataNode->changeValue().connect(this, );
}

WMainWindow::~WMainWindow()
{
	Wt::log("warning") << "stop WMainWindows";
}

void WMainWindow::setScene(std::shared_ptr<dyno::SceneGraph> scene)
{
	// try to stop the simulation
	controlContainer->stop();

	// setup scene graph
	mScene = scene;
	mSceneCanvas->setScene(mScene);
	controlContainer->setSceneGraph(mScene);
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

void WMainWindow::updateNodeGraphics()
{
	if (mNodeFlowWidget)
	{
		mNodeFlowWidget->update();
	}
	Wt::log("info") << "updateNodeGraphics!!!";
}

void WMainWindow::onKeyWentDown(const Wt::WKeyEvent& event)
{
	if (event.key() == Wt::Key::Delete || event.key() == Wt::Key::Backspace)
	{
		if (tab->currentIndex() == 0)
		{
			mNodeFlowWidget->onKeyWentDown();
		}
		else if (tab->currentIndex() == 1)
		{
			mModuleFlowWidget->onKeyWentDown();
		}
		else
		{
			return;
		}
	}
}

void WMainWindow::setInData(connectionData data, std::shared_ptr<dyno::Node> inNode)
{
	auto inPoint = data.inPoint;
	auto outNode = data.exportNode;
	auto outModule = data.exportModule;
	auto outPoint = data.outPoint;

	std::cout << outPoint.portIndex << std::endl;
	std::cout << inPoint.portIndex << std::endl;

	if (inPoint.portShape == PortShape::Diamond || inPoint.portShape == PortShape::Bullet)
	{
		outNode->connect(inNode->getImportNodes()[inPoint.portIndex]);
	}
	else if (inPoint.portShape == PortShape::Point)
	{
		dyno::FBase* field;

		if (outNode == nullptr && outModule != nullptr)
		{
			field = outModule->getOutputFields()[outPoint.portIndex - 1];
		}
		else
		{
			field = outNode->getOutputFields()[outPoint.portIndex - 1];
		}

		if (field != NULL)
		{
			dyno::FBase* inField;

			inField = inNode->getInputFields()[inPoint.portIndex];

			if (inField != nullptr)
				field->connect(inField);
		}
	}
}

void WMainWindow::initNavigationBar(Wt::WBorderLayout* layout)
{
	//create a navigation bar
	auto naviBar = layout->addWidget(std::make_unique<Wt::WNavigationBar>(), Wt::LayoutPosition::North);
	naviBar->addStyleClass("main-nav");
	naviBar->setResponsive(true);
	naviBar->setTitle("PeriDyno", "https://github.com/peridyno/peridyno");
	naviBar->setMargin(0);
}

void WMainWindow::initCenterContainer(Wt::WBorderLayout* layout)
{
	// create center
	auto centerContainer = layout->addWidget(std::make_unique<Wt::WContainerWidget>(), Wt::LayoutPosition::Center);
	centerContainer->setMargin(0);

	auto centerVbox = centerContainer->setLayout(std::make_unique<Wt::WVBoxLayout>());
	centerVbox->setContentsMargins(0, 0, 0, 0);

	// add scene canvas
	mSceneCanvas = centerVbox->addWidget(std::make_unique<WSimulationCanvas>());

	// add scene control
	controlContainer = centerVbox->addWidget(std::make_unique<WSimulationControl>());
	controlContainer->setSceneCanvas(mSceneCanvas);
}

void WMainWindow::initRightPanel(Wt::WContainerWidget* parent)
{
	// vertical layout
	auto layout = rightWidget->setLayout(std::make_unique<Wt::WVBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	parent->setMargin(0);

	auto widget0 = layout->addWidget(std::make_unique<Wt::WContainerWidget>());
	widget0->resize("100%", "100%");
	widget0->setMargin(0);
	tab = widget0->addNew<Wt::WTabWidget>();
	tab->resize("100%", "100%");
	tab->addTab(initNodeGraphics(), "NodeGraphics", Wt::ContentLoading::Eager);
	tab->addTab(initModuleGraphics(), "ModuleGraphics", Wt::ContentLoading::Eager);
	tab->addTab(initPython(), "Python", Wt::ContentLoading::Lazy);
	tab->addTab(initSample(), "Sample", Wt::ContentLoading::Lazy);
	tab->addTab(initSave(), "Save", Wt::ContentLoading::Lazy);
	tab->addTab(initLog(), "Log", Wt::ContentLoading::Lazy);
}

void WMainWindow::initAddNodePanel(Wt::WPanel* panel, AddNodeType addNodeType)
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

	// Recreating everything each time the scene switches affects performance.
	if (addNodeType == AddNodeType::NodeType)
	{
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
	}
	else if (addNodeType == AddNodeType::ModuleType)
	{
		auto nodeMap = dyno::Object::getClassMap();
		for (auto it = nodeMap->begin(); it != nodeMap->end(); ++it)
		{
			auto node_obj = dyno::Object::createObject(it->second->m_className);
			std::shared_ptr<dyno::Module> new_node(dynamic_cast<dyno::Module*>(node_obj));
			if (new_node == nullptr)
			{
				continue;
			}
			else
			{
				sp->addSuggestion(it->second->m_className);
			}
		}
	}

	auto name = layout3->addWidget(std::make_unique<Wt::WLineEdit>());

	addNodeType == AddNodeType::NodeType ? name->setPlaceholderText("Input Node Name") : name->setPlaceholderText("Input Module Name");

	sp->forEdit(name);

	auto addNodeButton = layout3->addWidget(std::make_unique<Wt::WPushButton>("Add"));

	addNodeButton->clicked().connect([=] {
		if (addNodeType == AddNodeType::NodeType)
		{
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
								mNodeFlowWidget->updateAll();
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
					mNodeFlowWidget->updateAll();
					name->setText("");
				}
			}
		}
		else
		{
			auto node_obj = dyno::Object::createObject(name->text().toUTF8());
			std::shared_ptr<dyno::Module> new_module(dynamic_cast<dyno::Module*>(node_obj));
			if (new_module != nullptr)
			{
				mModuleFlowWidget->addModule(new_module);
				new_module->setBlockCoord(Initial_x, Initial_y);
				Initial_x += 10;
				Initial_y += 10;
				mModuleFlowWidget->updateAll();
				name->setText("");
			}
		}
		});

	auto reorderNodeButton = layout3->addWidget(std::make_unique<Wt::WPushButton>("Reorder"));

	if (addNodeType == AddNodeType::NodeType)
	{
		reorderNodeButton->clicked().connect([=] {
			mNodeFlowWidget->reorderNode();
			});
	}
	else
	{
		reorderNodeButton->clicked().connect([=] {
			mModuleFlowWidget->reorderNode();
			});
	}
}

void WMainWindow::initPipelinePanel(Wt::WPanel* parent)
{
	auto widget = parent->setCentralWidget(std::make_unique<Wt::WContainerWidget>());
	auto layout = widget->setLayout(std::make_unique<Wt::WHBoxLayout>());

	layout->setContentsMargins(0, 0, 0, 0);
	auto ResetButton = layout->addWidget(std::make_unique<Wt::WPushButton>("Reset"));
	auto AnimationButton = layout->addWidget(std::make_unique<Wt::WPushButton>("Animation"));
	auto RenderingButton = layout->addWidget(std::make_unique<Wt::WPushButton>("Rendering"));

	// actions
	ResetButton->clicked().connect([=] {
		mModuleFlowWidget->showResetPipeline();
		});
	AnimationButton->clicked().connect([=] {
		mModuleFlowWidget->showAnimationPipeline();
		});
	RenderingButton->clicked().connect([=] {
		mModuleFlowWidget->showGraphicsPipeline();
		});
}

std::unique_ptr<Wt::WWidget> WMainWindow::initNodeGraphics()
{
	nodeGraphicsWidget = std::make_unique<WNodeGraphics>();
	initAddNodePanel(nodeGraphicsWidget->addPanel);

	if (mScene)
	{
		auto painteContainer = nodeGraphicsWidget->nodePanel->setCentralWidget(std::make_unique<Wt::WContainerWidget>());
		painteContainer->setMargin(0);
		mNodeFlowWidget = painteContainer->addWidget(std::make_unique<WtNodeFlowWidget>(mScene));
		mNodeFlowWidget->resize(viewportWidth * WIDTH_SCALE, viewportHeight * 0.4);
	}

	// Parameter list
	auto parameterWidget = nodeGraphicsWidget->layout->addWidget(std::make_unique<Wt::WContainerWidget>());

	auto promptNodeWidget = nodeGraphicsWidget->layout->addWidget(std::make_unique<Wt::WContainerWidget>());

	//action for selection change
	mNodeFlowWidget->selectNodeSignal().connect([=](int selectNum)
		{
			if (selectNum > 0)
			{
				for (auto it = mScene->begin(); it != mScene->end(); it++)
				{
					auto m = it.get();
					if (m->objectId() == selectNum)
					{
						mParameterDataNode->setNode(m);
						mParameterDataNode->createParameterPanel(parameterWidget);
						mSceneCanvas->selectNode(m);
						mModuleFlowWidget->setNode(m);
						mSceneCanvas->update();
					}
				}
			}
		});

	mNodeFlowWidget->updateCanvas().connect([=]()
		{
			this->updateCanvas();
		});

	mNodeFlowWidget->prompt().connect([=](std::map<std::string, connectionData> promptNodes)
		{
			mPromptPanel->createPromptPanel(promptNodeWidget, promptNodes);
		});

	if (mSceneCanvas)
	{
		mSceneCanvas->selectNodeSignal().connect([=](std::shared_ptr<dyno::Node> node)
			{
				if (node != nullptr)
					mNodeFlowWidget->setSelectNode(node);
			});
	}

	mPromptPanel->addPromptNode().connect([=](connectionData addNodeData)
		{
			if (addNodeData.inportNode != nullptr)
			{
				auto node_obj = dyno::Object::createObject(addNodeData.inportNode->caption());
				std::shared_ptr<dyno::Node> new_node(dynamic_cast<dyno::Node*>(node_obj));
				mScene->addNode(new_node);
				//new_node->setBlockCoord(Initial_x, Initial_y);

				setInData(addNodeData, new_node);

				Initial_x += 10;
				Initial_y += 10;
				mNodeFlowWidget->reorderNode();
				mNodeFlowWidget->updateAll();
			}
			if (addNodeData.inportModule != nullptr)
			{
				auto node_obj = dyno::Object::createObject(addNodeData.inportModule->caption());
				std::shared_ptr<dyno::Module> new_node(dynamic_cast<dyno::Module*>(node_obj));
				mModuleFlowWidget->addModule(new_node);
				//new_node->setBlockCoord(Initial_x, Initial_y);
				Initial_x += 10;
				Initial_y += 10;
				mModuleFlowWidget->reorderNode();
				mModuleFlowWidget->updateAll();
			}
		});

	return std::move(nodeGraphicsWidget);
}

std::unique_ptr<Wt::WWidget> WMainWindow::initModuleGraphics()
{
	moduleGraphicsWidget = std::make_unique<WModuleGraphics>();
	initAddNodePanel(moduleGraphicsWidget->addPanel, AddNodeType::ModuleType);
	initPipelinePanel(moduleGraphicsWidget->pipelinePanel);

	if (mScene)
	{
		auto painteContainer = moduleGraphicsWidget->modulePanel->setCentralWidget(std::make_unique<Wt::WContainerWidget>());
		painteContainer->setMargin(0);
		mModuleFlowWidget = painteContainer->addWidget(std::make_unique<WtModuleFlowWidget>(mScene));
		mModuleFlowWidget->resize(viewportWidth * WIDTH_SCALE, viewportHeight * 0.4);
	}

	// Parameter list
	auto parameterWidget = moduleGraphicsWidget->layout->addWidget(std::make_unique<Wt::WContainerWidget>());

	auto promptNodeWidget = moduleGraphicsWidget->layout->addWidget(std::make_unique<Wt::WContainerWidget>());

	//action for selection change
	mModuleFlowWidget->selectModuleSignal().connect([=](std::shared_ptr<dyno::Module> module)
		{
			if (module != nullptr)
			{
				mParameterDataNode->setModule(module);
				mParameterDataNode->createParameterPanelModule(parameterWidget);
			}
		});

	mModuleFlowWidget->prompt().connect([=](std::map<std::string, connectionData> promptModules)
		{
			mPromptPanel->createPromptPanel(promptNodeWidget, promptModules);
		});

	return std::move(moduleGraphicsWidget);
}

std::unique_ptr<Wt::WWidget> WMainWindow::initPython()
{
	pythonWidget->resize(viewportWidth * WIDTH_SCALE, viewportHeight * HEIGHT_SCALE);

	pythonWidget->updateSceneGraph().connect([=](std::shared_ptr<dyno::SceneGraph> scene) {
		if (scene)
		{
			setScene(scene);

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
	auto saveWidget = std::make_unique<WSaveWidget>(this, viewportWidth * WIDTH_SCALE / 2);
	saveWidget->setMargin(10);

	Wt::WApplication::instance()->styleSheet().addRule(
		".save-middle",
		"justify-items: anchor-center;"
	);

	saveWidget->setStyleClass("save-middle");

	return saveWidget;
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