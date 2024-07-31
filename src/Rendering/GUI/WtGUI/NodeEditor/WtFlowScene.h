#pragma once

#include <Wt/WPaintDevice.h>
#include <Wt/WPaintedWidget.h>
#include <Wt/WPainter.h>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/unordered_map.hpp>

#include "Export.hpp"
#include "WtNode.h"
#include "WtDataModelRegistry.h"
#include <memory>

class WtNodeDataModel;
//class FlowItemInterface;
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

	WtNode& createNode(std::unique_ptr<WtNodeDataModel>&& dataModel);

public:

	std::unordered_map<boost::uuids::uuid, std::unique_ptr<WtNode> > const& nodes() const;

	std::unordered_map<boost::uuids::uuid, std::shared_ptr<WtConnection> > const& connections() const;

	std::vector<WtNode*> allNodes() const;

	std::vector<WtNode*> selectedNodes() const;

public:

	void clearScene();

	void save() const;

	void load();

private:

	using SharedConnection = std::shared_ptr<WtConnection>;
	using UniqueNode = std::unique_ptr<WtNode>;

	// DO NOT reorder this member to go after the others.
	// This should outlive all the connections and nodes of
	// the graph, so that nodes can potentially have pointers into it,
	// which is why it comes first in the class.
	std::shared_ptr<WtDataModelRegistry> _registry;

	boost::unordered_map<boost::uuids::uuid, SharedConnection> _connections;
	boost::unordered_map<boost::uuids::uuid, UniqueNode>       _nodes;
};