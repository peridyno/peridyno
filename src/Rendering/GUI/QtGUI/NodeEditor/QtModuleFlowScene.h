#pragma once

#include "nodes/QFlowScene"

#include "Node.h"
#include "QtNodeWidget.h"

using dyno::Node;


namespace Qt
{
	/// Scene holds connections and nodes.
	class QtModuleFlowScene
		: public QtFlowScene
	{
		Q_OBJECT
	public:

		QtModuleFlowScene(std::shared_ptr<QtDataModelRegistry> registry,
			QObject* parent = Q_NULLPTR);

		QtModuleFlowScene(QObject* parent = Q_NULLPTR, QtNodeWidget* node_widget = nullptr);


		~QtModuleFlowScene();

	public:
		// push and refresh modules to parent_node's graphicsPipeline
		void pushModule();


	public Q_SLOTS:
		void showNodeFlow(Node* node);
		void moveModulePosition(QtNode& n, const QPointF& newLocation);


	private:
		std::weak_ptr<dyno::Node> m_node;
		QtNodeWidget* m_parent_node;
	};
}
