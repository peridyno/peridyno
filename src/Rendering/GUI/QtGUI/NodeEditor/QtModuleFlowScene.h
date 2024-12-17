#pragma once

#include "nodes/QFlowScene"

#include "Node.h"
#include "QtNodeWidget.h"

using dyno::Node;
using dyno::Module;
using dyno::Pipeline;

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

	Q_SIGNALS:
		void nodeExportChanged();

	public Q_SLOTS:
		void showModuleFlow(Node* node);

		void updateModuleGraphView();

		void reorderAllModules();

		void addModule(QtNode& n);

		void deleteModule(QtNode& n);

		void moveModule(QtNode& n, const QPointF& newLocation);

		void showResetPipeline();

		void showAnimationPipeline();

		void showGraphicsPipeline();

		/**
		 * pos: screen pos
		 */
		void promoteOutput(QtNode& n, const PortIndex index, const QPointF& pos);

	private:
		std::shared_ptr<dyno::Node> mNode;
		std::shared_ptr<dyno::Pipeline> mActivePipeline;

		//A virtual module to store all state variables
		std::shared_ptr<dyno::Module> mStates = nullptr;

		float mDx = 100.0f;
		float mDy = 50.0f;

		bool mReorderResetPipeline = true;
		bool mReorderGraphicsPipeline = true;

		bool mEditingEnabled = true;
	};
}
