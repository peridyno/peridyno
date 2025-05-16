#pragma once

#include "WtNode.h"
#include "WtConnection.h"
#include "WtNodeFlowScene.h"

class WtDataModelRegistry;
class WtNodeFlowScene;
class WtNodeDataModel;

class WtInteraction
{
public:
	WtInteraction(WtNode& node, WtConnection& connection, WtFlowScene& scene, connectionPointData inPoint, connectionPointData outPoint, std::shared_ptr<Node> inNode, std::shared_ptr<Node> outNode);

	WtInteraction(WtNode& node, WtConnection& connection, connectionPointData inPoint, connectionPointData outPoint, std::shared_ptr<dyno::Module> inModule, std::shared_ptr<dyno::Module> outModule);

	bool canConnect(PortIndex& portIndex, TypeConverter& converter);

	bool tryConnect();
private:

	PortType connectionRequiredPort() const;

	bool isNodePortAccessible(PortType portType, PortIndex portIndex) const;

	void setInData(PortIndex portIndex);

private:

	WtNode* _node;

	WtConnection* _connection;

	WtFlowScene* _scene;

	connectionPointData _inPoint;

	connectionPointData _outPoint;

	std::shared_ptr<Node> _inNode;

	std::shared_ptr<Node> _outNode;

	std::shared_ptr<dyno::Module> _inModule;

	std::shared_ptr<dyno::Module> _outModule;
};