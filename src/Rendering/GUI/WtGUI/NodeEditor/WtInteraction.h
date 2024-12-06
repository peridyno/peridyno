#pragma once

#include "WtNode.h"
#include "WtConnection.h"
#include "WtNode.h"
#include "WtNodeFlowScene.h"

class WtDataModelRegistry;
class WtNodeFlowScene;
class WtNodeDataModel;

class WtInteraction
{
public:
	WtInteraction(WtNode& node, WtConnection& connection, WtFlowScene& scene, connectionPointData inPoint);

	bool canConnect(PortIndex& portIndex, TypeConverter& converter);

	bool tryConnect();

	bool disconnect(PortType portToDisconnect);

private:

	PortType connectionRequiredPort() const;

	bool isNodePortAccessible(PortType portType, PortIndex portIndex) const;

private:

	WtNode* _node;

	WtConnection* _connection;

	WtFlowScene* _scene;

	connectionPointData _inPoint;
};