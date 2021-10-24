#pragma once

#include "QtFlowScene.h"

#include "Node.h"
#include "Nodes/QtNodeWidget.h"

namespace QtNodes
{

using dyno::Node;

/// Scene holds connections and nodes.
class NODE_EDITOR_PUBLIC QtModuleFlowScene
  : public QtFlowScene
{
	Q_OBJECT
public:

	QtModuleFlowScene(std::shared_ptr<DataModelRegistry> registry,
			QObject * parent = Q_NULLPTR);

	QtModuleFlowScene(QObject * parent = Q_NULLPTR, QtNodeWidget* node_widget = nullptr);


	~QtModuleFlowScene();

public:
	// push and refresh modules to parent_node's graphicsPipeline
	void pushModule();


public Q_SLOTS:
	void showNodeFlow(Node* node);
	void moveModulePosition(QtBlock& n, const QPointF& newLocation);


private:
	std::weak_ptr<dyno::Node> m_node;
	QtNodeWidget* m_parent_node;
};

}
