#include "WSceneDataModel.h"
#include <SceneGraph.h>

WNodeDataModel::WNodeDataModel()
{
}

void WNodeDataModel::setScene(std::shared_ptr<dyno::SceneGraph> scene)
{
	mScene = scene;

	layoutAboutToBeChanged().emit();

	for (NodeItem* item : mNodeList)
		delete item;
	mNodeList.clear();

	if (mScene)
	{
		for (auto node = scene->begin(); node != scene->end(); node++)
		{
			NodeItem* item = new NodeItem;
			item->id = mNodeList.size();
			item->ref = node.get();
			item->parent = 0;
			mNodeList.push_back(item);
		}
	}
	layoutChanged().emit();
}

Wt::WModelIndex WNodeDataModel::parent(const Wt::WModelIndex& index) const
{
	return Wt::WModelIndex();
}

Wt::WModelIndex WNodeDataModel::index(int row, int column, const Wt::WModelIndex& parent) const
{
	if (parent.isValid())
	{
		return Wt::WModelIndex();
	}
	return createIndex(row, column, row);
}

int WNodeDataModel::columnCount(const Wt::WModelIndex& parent) const
{
	return 2;
}

int WNodeDataModel::rowCount(const Wt::WModelIndex& parent) const
{
	if (parent.isValid())
		return 0;
	return mNodeList.size();
}

Wt::cpp17::any WNodeDataModel::data(const Wt::WModelIndex& index, Wt::ItemDataRole role) const
{
	if (index.isValid())
	{
		auto node = mNodeList[index.internalId()]->ref;

		if (role == Wt::ItemDataRole::Display || role == Wt::ItemDataRole::ToolTip)
		{
			if (index.column() == 0)
			{
				return node->getName();
			}
			if (index.column() == 1)
			{
				return node->getClassInfo()->getClassName();
			}
		}
		else if (role == Wt::ItemDataRole::Decoration)
		{
			if (index.column() == 0)
			{
				if (node->getName() == "cube")
				{
					return "icons/cube.png";
				}
				else if (node->getName() == "Mesh")
				{
					return "icons/mesh.png";
				}
				return std::string("icons/node.png");
			}
		}
	}
	return Wt::cpp17::any();
}

Wt::cpp17::any WNodeDataModel::headerData(int section, Wt::Orientation orientation, Wt::ItemDataRole role) const
{
	if (orientation == Wt::Orientation::Horizontal && role == Wt::ItemDataRole::Display) {
		switch (section) {
		case 0:
			return std::string("Node");
		case 1:
			return std::string("Type");
		default:
			return Wt::cpp17::any();
		}
	}
	else
		return Wt::cpp17::any();
}

std::shared_ptr<dyno::Node> WNodeDataModel::getNode(const Wt::WModelIndex& index)
{
	return mNodeList[index.internalId()]->ref;
}

void WModuleDataModel::setNode(std::shared_ptr<dyno::Node> node)
{
	mNode = node;
	layoutAboutToBeChanged().emit();
	layoutChanged().emit();
}

int WModuleDataModel::columnCount(const Wt::WModelIndex& parent) const
{
	return 2;
}

int WModuleDataModel::rowCount(const Wt::WModelIndex& parent) const
{
	if (mNode != 0)
	{
		return mNode->getModuleList().size();
	}
	return 0;
}

Wt::cpp17::any WModuleDataModel::data(const Wt::WModelIndex& index, Wt::ItemDataRole role) const
{
	if (mNode != 0 && index.isValid())
	{
		auto mod = mNode->getModuleList();
		auto iter = mod.begin();
		std::advance(iter, index.row());

		if (role == Wt::ItemDataRole::Display || role == Wt::ItemDataRole::ToolTip)
		{
			if (index.column() == 0)
			{
				return (*iter)->getName();
			}
			if (index.column() == 1)
			{
				return (*iter)->getModuleType();
			}
		}
		else if (role == Wt::ItemDataRole::Decoration)
		{
			if (index.column() == 0)
			{
				return std::string("icons/module.png");
			}
		}
	}
	return Wt::cpp17::any();
}

Wt::cpp17::any WModuleDataModel::headerData(int section, Wt::Orientation orientation, Wt::ItemDataRole role) const
{
	if (orientation == Wt::Orientation::Horizontal && role == Wt::ItemDataRole::Display) {
		switch (section) {
		case 0:
			return std::string("Module");
		case 1:
			return std::string("Type");
		default:
			return Wt::cpp17::any();
		}
	}
	else
		return Wt::cpp17::any();
}

std::shared_ptr<dyno::Module> WModuleDataModel::getModule(const Wt::WModelIndex& index)
{
	if (mNode != 0 && index.isValid())
	{
		auto mod = mNode->getModuleList();
		auto iter = mod.begin();
		std::advance(iter, index.row());
		return *iter;
	}
	return std::shared_ptr<dyno::Module>();
}

WPromptNode::WPromptNode(std::map<std::string, std::tuple<std::string, int>> promptNodes)
{
	setPromptNode(promptNodes);
}

WPromptNode::~WPromptNode()
{
}

void WPromptNode::setPromptNode(std::map<std::string, std::tuple<std::string, int>> promptNodes)
{
	mPromptNodes = promptNodes;
	NodeItem* item = new NodeItem;
	item->index = 1;
	item->parentIndex = 0;
	item->name = "root";
	mNodeList.push_back(item);
	for (int i = 0; i < 10; i++)
	{
		NodeItem* item = new NodeItem;
		item->index = i + 2;
		item->parentIndex = 1;
		item->name = "tesst";
		mNodeList.push_back(item);
	}
	//if (!mPromptNodes.empty())
	//{
	//	std::map<std::string, std::vector<NodeItem*>> promptNodesSort;
	//	int i = 0;
	//	int j = 0;

	//	for (auto promptNode : mPromptNodes)
	//	{
	//		NodeItem* item = new NodeItem;
	//		auto temp = promptNode.second;
	//		item->name = promptNode.first;
	//		item->type = std::get<0>(temp);
	//		item->connectIndex = std::get<1>(temp);
	//		item->index = i;
	//		item->parentIndex = j;

	//		if (promptNodesSort.find(std::get<0>(temp)) != promptNodesSort.end())
	//		{
	//			promptNodesSort.find(std::get<0>(temp))->second.push_back(item);
	//			i++;
	//		}
	//		else
	//		{
	//			std::vector<NodeItem*> newVector;
	//			j++;
	//			item->parentIndex = j;
	//			i = 0;
	//			newVector.push_back(item);
	//			promptNodesSort.emplace(std::get<0>(temp), newVector);
	//		}
	//	}

	//	for (auto nodeSort : promptNodesSort)
	//	{
	//		mNodeList.insert(mNodeList.end(), nodeSort.second.begin(), nodeSort.second.end());
	//	}
	//}
}

Wt::WModelIndex WPromptNode::parent(const Wt::WModelIndex& index) const
{
	if (!index.isValid())
	{
		std::cout << "0" << std::endl;
		return Wt::WModelIndex();
	}
	else if(index.internalId() == 0)
	{
		std::cout << "1" << std::endl;
		return Wt::WModelIndex();
	}
	else
	{
		std::cout << "test" << std::endl;
		auto item = mNodeList[index.internalId()];
		return createIndex(item->index, 0, item->parentIndex);
	}
}

Wt::WModelIndex WPromptNode::index(int row, int column, const Wt::WModelIndex& parent) const
{
	int parentId;
	if (!parent.isValid())
	{
		parentId = 0;
	}
	else
	{
		int grandParentId = parent.internalId();
		parentId = mNodeList[grandParentId]->parentIndex;
	}
	return createIndex(row, column, parentId);
}

int WPromptNode::columnCount(const Wt::WModelIndex& parent) const
{
	return 2;
}

int WPromptNode::rowCount(const Wt::WModelIndex& parent) const
{
	if (parent.isValid())
		return 0;
	return mNodeList.size();
}

Wt::cpp17::any WPromptNode::data(const Wt::WModelIndex& index, Wt::ItemDataRole role) const
{
	if (index.isValid())
	{
		auto data = mNodeList[index.internalId()];

		if (role == Wt::ItemDataRole::Display)
		{
			if (index.column() == 0)
			{
				return data->type;
			}
			if (index.column() == 1)
			{
				return data->name;
			}
		}
	}
	return Wt::cpp17::any();
}

Wt::cpp17::any WPromptNode::headerData(int section, Wt::Orientation orientation, Wt::ItemDataRole role) const
{
	if (orientation == Wt::Orientation::Horizontal && role == Wt::ItemDataRole::Display) {
		switch (section) {
		case 0:
			return std::string("Type");
		case 1:
			return std::string("Name");
		default:
			return Wt::cpp17::any();
		}
	}
	else
		return Wt::cpp17::any();
}