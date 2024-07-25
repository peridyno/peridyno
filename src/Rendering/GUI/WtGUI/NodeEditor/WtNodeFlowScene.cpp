#include "WtNodeFlowScene.h"

WtNodeFlowScene::WtNodeFlowScene(Wt::WPainter* painter)
{
	auto classMap = dyno::Object::getClassMap();
	auto ret = std::make_shared<WtDataModelRegistry>();
	int id = 0;
	for (auto const c : *classMap)
	{
		id++;

		std::string str = c.first;
		auto obj = dyno::Object::createObject(str);
		std::shared_ptr<dyno::Node> node(dynamic_cast<dyno::Node*>(obj));

		if (node != nullptr)
		{
			WtDataModelRegistry::RegistryItemCreator creator = [str]()
				{
					auto node_obj = dyno::Object::createObject(str);
					std::shared_ptr<dyno::Node> new_node(dynamic_cast<dyno::Node*>(node_obj));
					auto dat = std::make_unique<WtNodeWidget>(std::move(new_node));
					return dat;
				};

			std::string category = node->getNodeType();
			ret->registerModel<WtNodeWidget>(category, creator);
		}
	}
}

WtNodeFlowScene::~WtNodeFlowScene() {}