#pragma once

#include <Wt/WPaintDevice.h>
#include <Wt/WPaintedWidget.h>
#include <Wt/WPainter.h>
#include "guid.hpp"

#include "Export.hpp"
#include "WtNode.h"
#include "WtDataModelRegistry.h"
#include "WtConnectionGraphicsObject.h"
#include "WtConnection.h"
#include <memory>

class WtNodeDataModel;
class WtNode;
class WtConnection;
class WtNodeGraphicsObject;
class WtConnectionGraphicsObject;
class WtNodeStyle;

class WtFlowScene
{
public:

	WtFlowScene();

	WtFlowScene(std::shared_ptr<WtDataModelRegistry> registry);

	~WtFlowScene();

public:

	std::shared_ptr<WtConnection> createConnection(PortType connectedPort,
		WtNode& node,
		PortIndex portIndex,
		Wt::WPainter* painter);

	std::shared_ptr<WtConnection> createConnection(WtNode& nodeIn,
		PortIndex portIndexIn,
		WtNode& nodeOut,
		PortIndex portIndexOut,
		Wt::WPainter* painter,
		TypeConverter const& converter = TypeConverter{});

	//std::shared_ptr<WtConnection> restoreConnection(QJsonObject const& connectionJson);

	void deleteConnection(WtConnection& connection);

	WtNode& createNode(std::unique_ptr<WtNodeDataModel>&& dataModel, Wt::WPainter* painter, bool isSelected);

	//QtNode& restoreNode(QJsonObject const& nodeJson);

	WtDataModelRegistry& registry() const;

	void setRegistry(std::shared_ptr<WtDataModelRegistry> registry);

	void iterateOverNodes(std::function<void(WtNode*)> const& visitor);

	void iterateOverNodeData(std::function<void(WtNodeDataModel*)> const& visitor);

	void iterateOverNodeDataDependentOrder(std::function<void(WtNodeDataModel*)> const& visitor);

	Wt::WPointF getNodePosition(WtNode const& node) const;

	void setNodePosition(WtNode& node, Wt::WPointF const& pos) const;

	//QSizeF getNodeSize(WtNode const& node) const;

	void removeNode(WtNode& node);

	void clearNode(WtNode& node);

public:

	std::unordered_map<Wt::Guid, std::unique_ptr<WtNode> > const& nodes() const;

	std::unordered_map<Wt::Guid, std::shared_ptr<WtConnection> > const& connections() const;

	std::vector<WtNode*> allNodes() const;

	std::vector<WtNode*> selectedNodes() const;

public:

	void clearScene();

	void save() const;

	void load();

	//QByteArray saveToMemory() const;

	//void loadFromMemory(const QByteArray& data);

private:

	using SharedConnection = std::shared_ptr<WtConnection>;
	using UniqueNode = std::unique_ptr<WtNode>;

	// DO NOT reorder this member to go after the others.
	// This should outlive all the connections and nodes of
	// the graph, so that nodes can potentially have pointers into it,
	// which is why it comes first in the class.
	std::shared_ptr<WtDataModelRegistry> _registry;

	std::unordered_map<Wt::Guid, SharedConnection> _connections;
	std::unordered_map<Wt::Guid, UniqueNode>       _nodes;
};