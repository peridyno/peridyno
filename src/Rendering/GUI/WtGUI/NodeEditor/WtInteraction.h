#pragma once

#include "WtNode.h"
#include "WtConnection.h"

class WtDataModelRegistry;
class WtFlowScene;
class WtNodeDataModel;

class WtInteraction
{
public:
	WtInteraction(WtNode& node, WtConnection& connection, WtFlowScene& scene);

	bool canConnect(PortIndex& portIndex, TypeConverter& converter) const;

	bool tryConnect() const;

	bool disconnect(PortType portToDisconnect) const;

private:

	PortType connectionRequiredPort() const;

	Wt::WPointF connectionEndScenePosition(PortType) const;

	Wt::WPointF nodePortScenePosition(PortType portType, PortIndex portIndex) const;

	PortIndex nodePortIndexUnderScenePoint(PortType portType, Wt::WPointF const& p) const;

	bool isNodePortAccessible(PortType portType, PortIndex portIndex) const;

private:

	WtNode* _node;

	WtConnection* _connection;

	WtFlowScene* _scene;
};