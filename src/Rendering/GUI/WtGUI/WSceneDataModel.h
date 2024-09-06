#pragma once

#include <Wt/WAbstractItemModel.h>
#include <Wt/WAbstractTableModel.h>
#include <Wt/WText.h>
#include <Wt/WPanel.h>
#include <Wt/WTable.h>
#include <Wt/WDoubleSpinBox.h>
#include <Wt/WLogger.h>

#include <FBase.h>

namespace dyno
{
	class Node;
	class Module;
	class SceneGraph;
};

class WNodeDataModel : public Wt::WAbstractItemModel
{
public:
	WNodeDataModel();

	void setScene(std::shared_ptr<dyno::SceneGraph> scene);

	virtual Wt::WModelIndex parent(const Wt::WModelIndex& index) const;
	virtual Wt::WModelIndex index(int row, int column,
		const Wt::WModelIndex& parent = Wt::WModelIndex()) const;

	virtual int columnCount(const Wt::WModelIndex& parent = Wt::WModelIndex()) const;
	virtual int rowCount(const Wt::WModelIndex& parent = Wt::WModelIndex()) const;

	virtual Wt::cpp17::any data(const Wt::WModelIndex& index,
		Wt::ItemDataRole role = Wt::ItemDataRole::Display) const;

	virtual Wt::cpp17::any headerData(int section,
		Wt::Orientation orientation = Wt::Orientation::Horizontal,
		Wt::ItemDataRole role = Wt::ItemDataRole::Display) const;

	std::shared_ptr<dyno::Node> getNode(const Wt::WModelIndex& index);

private:
	std::shared_ptr<dyno::SceneGraph> mScene;

	struct NodeItem
	{
		int id = -1;
		int offset = 0;

		NodeItem* parent;
		std::vector<NodeItem*>	children;

		std::shared_ptr<dyno::Node> ref;
	};

	std::vector<NodeItem*> mNodeList;
};


class WModuleDataModel : public Wt::WAbstractTableModel
{
public:

	void setNode(std::shared_ptr<dyno::Node> node);

	virtual int columnCount(const Wt::WModelIndex& parent = Wt::WModelIndex()) const;
	virtual int rowCount(const Wt::WModelIndex& parent = Wt::WModelIndex()) const;

	virtual Wt::cpp17::any data(const Wt::WModelIndex& index,
		Wt::ItemDataRole role = Wt::ItemDataRole::Display) const;

	virtual Wt::cpp17::any headerData(int section,
		Wt::Orientation orientation = Wt::Orientation::Horizontal,
		Wt::ItemDataRole role = Wt::ItemDataRole::Display) const;

	std::shared_ptr<dyno::Module> getModule(const Wt::WModelIndex& index);
private:
	std::shared_ptr<dyno::Node> mNode;
};

