#include "WNodeGraphics.h"


WNodeGraphics::WNodeGraphics()
{
	auto layout = this->setLayout(std::make_unique<Wt::WVBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	this->setMargin(0);

	// add node
	addPanel = layout->addWidget(std::make_unique<Wt::WPanel>());
	addPanel->setTitle("Add Node");
	addPanel->setCollapsible(false);
	//initAddNodePanel(addPanel);

	// node graphics
	nodePanel = layout->addWidget(std::make_unique<Wt::WPanel>());
	nodePanel->setTitleBar(false);
	nodePanel->setCollapsible(false);
	nodePanel->setMargin(0);

	// Parameter list
	parameterPanel = layout->addWidget(std::make_unique<Wt::WPanel>());
	parameterPanel->setTitle("Control Variable");
	parameterPanel->setCollapsible(true);
	parameterPanel->setStyleClass("scrollable-content");
}

WNodeGraphics::~WNodeGraphics() {}


//void WNodeGraphics::initAddNodePanel(Wt::WPanel* panel)
//{
//	auto widget3 = panel->setCentralWidget(std::make_unique<Wt::WContainerWidget>());
//	auto layout3 = widget3->setLayout(std::make_unique<Wt::WHBoxLayout>());
//	layout3->setContentsMargins(0, 0, 0, 0);
//
//	Wt::WSuggestionPopup::Options nodeOptions;
//	nodeOptions.highlightBeginTag = "<span class=\"highlight\">";
//	nodeOptions.highlightEndTag = "</span>";
//
//	Wt::WSuggestionPopup* sp = layout3->addChild(std::make_unique<Wt::WSuggestionPopup>(
//		Wt::WSuggestionPopup::generateMatcherJS(nodeOptions),
//		Wt::WSuggestionPopup::generateReplacerJS(nodeOptions)
//	));
//
//	auto& pages = dyno::NodeFactory::instance()->nodePages();
//	for (auto iPage = pages.begin(); iPage != pages.end(); iPage++)
//	{
//		auto& groups = iPage->second->groups();
//		{
//			for (auto iGroup = groups.begin(); iGroup != groups.end(); iGroup++)
//			{
//				auto& actions = iGroup->second->actions();
//				for (auto action : actions)
//				{
//					sp->addSuggestion(action->caption());
//				}
//			}
//		}
//	}
//
//	auto nodeMap = dyno::Object::getClassMap();
//	for (auto it = nodeMap->begin(); it != nodeMap->end(); ++it)
//	{
//		auto node_obj = dyno::Object::createObject(it->second->m_className);
//		std::shared_ptr<dyno::Node> new_node(dynamic_cast<dyno::Node*>(node_obj));
//		if (new_node == nullptr)
//		{
//			continue;
//		}
//		else
//		{
//			sp->addSuggestion(it->second->m_className);
//		}
//	}
//	auto name = layout3->addWidget(std::make_unique<Wt::WLineEdit>());
//	name->setPlaceholderText("node name");
//
//	sp->forEdit(name);
//
//	auto addNodeButton = layout3->addWidget(std::make_unique<Wt::WPushButton>("Add"));
//
//	addNodeButton->clicked().connect([=] {
//		bool flag = true;
//
//		for (auto iPage = pages.begin(); iPage != pages.end(); iPage++)
//		{
//			auto& groups = iPage->second->groups();
//			{
//				for (auto iGroup = groups.begin(); iGroup != groups.end(); iGroup++)
//				{
//					auto& actions = iGroup->second->actions();
//					for (auto action : actions)
//					{
//						if (action->caption() == name->text().toUTF8())
//						{
//							auto new_node = mScene->addNode(action->action()());
//							new_node->setBlockCoord(Initial_x, Initial_y);
//							Initial_x += 20;
//							Initial_y += 20;
//							name->setText("");
//							mFlowWidget->updateForAddNode();
//							mNodeDataModel->setScene(mScene);
//							flag = false;
//						}
//					}
//				}
//			}
//		}
//
//		if (flag)
//		{
//			auto node_obj = dyno::Object::createObject(name->text().toUTF8());
//			std::shared_ptr<dyno::Node> new_node(dynamic_cast<dyno::Node*>(node_obj));
//			if (new_node != nullptr)
//			{
//				mScene->addNode(new_node);
//				new_node->setBlockCoord(Initial_x, Initial_y);
//				Initial_x += 10;
//				Initial_y += 10;
//				std::cout << Initial_x << std::endl;
//				std::cout << "!!!!!!!!!!!" << std::endl;
//				mFlowWidget->updateForAddNode();
//				mNodeDataModel->setScene(mScene);
//				name->setText("");
//			}
//
//		}
//		});
//
//	auto reorderNodeButton = layout3->addWidget(std::make_unique<Wt::WPushButton>("Reorder"));
//
//	reorderNodeButton->clicked().connect([=] {
//		mFlowWidget->reorderNode();
//		});
//}