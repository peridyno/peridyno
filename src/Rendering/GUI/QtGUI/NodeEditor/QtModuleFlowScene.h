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

		QtModuleFlowScene(QObject* parent = Q_NULLPTR, QtNodeWidget* widget = nullptr);


		~QtModuleFlowScene();

	public:
		void enableEditing();
		void disableEditing();

	public Q_SLOTS:
		void showModuleFlow(Node* node);

		void updateModuleGraphView();

		void reorderAllModules();

		void addModule(QtNode& n);

		void deleteModule(QtNode& n);

		void moveModule(QtNode& n, const QPointF& newLocation);
	private:
		std::shared_ptr<dyno::Node> mNode;

		bool mEditingEnabled = true;

		float mDx = 100.0f;
		float mDy = 50.0f;
	};
}
