#pragma once

#include "nodes/QFlowScene"

#include "Topology/MaterialManager.h"
#include "QtNodeWidget.h"

using dyno::Node;
using dyno::Module;
using dyno::Pipeline;
using dyno::CustomMaterial;

namespace Qt
{
	/// Scene holds connections and nodes.
	class QtMaterialFlowScene
		: public QtFlowScene
	{
		Q_OBJECT
	public:

		QtMaterialFlowScene(std::shared_ptr<QtDataModelRegistry> registry,
			QObject* parent = Q_NULLPTR);

		QtMaterialFlowScene(std::shared_ptr<CustomMaterial> customMaterial, QObject* parent = Q_NULLPTR);


		~QtMaterialFlowScene();

	public:
		void enableEditing();
		void disableEditing();

	Q_SIGNALS:
		void nodeExportChanged();

	public Q_SLOTS:
		void showMaterialFlow(std::shared_ptr<CustomMaterial> customMaterial);

		void updateModuleGraphView();

		void reorderAllModules();

		void addModule(QtNode& n);

		void deleteModule(QtNode& n);

		void moveModule(QtNode& n, const QPointF& newLocation);

		void showCustomMaterialPipeline();

		void reconstructActivePipeline();

		/**
		 * pos: screen pos
		 */
		void promoteOutput(QtNode& n, const PortIndex index, const QPointF& pos);

	private:

		std::shared_ptr<dyno::CustomMaterial> mCustomMaterial;
		std::shared_ptr<dyno::MaterialPipeline> mMaterialPipline;


		float mDx = 100.0f;
		float mDy = 50.0f;

		bool mReorderResetPipeline = true;
		bool mReorderGraphicsPipeline = true;

		bool mEditingEnabled = true;
	};
}
