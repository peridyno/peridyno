#pragma once

#include <Wt/WPaintDevice.h>
#include <Wt/WPaintedWidget.h>
#include <Wt/WPainter.h>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "Export.hpp"
#include "WtNode.h"
#include "WtDataModelRegistry.h"

class WtFlowScene
{
public:
	WtFlowScene();
	WtFlowScene(std::shared_ptr<WtDataModelRegistry> registry, Wt::WPainter* painter);
	~WtFlowScene();

public:

	WtNode& createNode(std::unique_ptr<WtNodeDataModel>&& dataModel);

private:
	using SharedConnection = std::shared_ptr<WtConnection>;
	using UniqueNode = std::unique_ptr<WtNode>;

	// DO NOT reorder this member to go after the others.
	// This should outlive all the connections and nodes of
	// the graph, so that nodes can potentially have pointers into it,
	// which is why it comes first in the class.
	std::shared_ptr<WtDataModelRegistry> _registry;

	std::unordered_map<boost::uuids::uuid, SharedConnection> _connections;
	std::unordered_map<boost::uuids::uuid, UniqueNode>       _nodes;
};