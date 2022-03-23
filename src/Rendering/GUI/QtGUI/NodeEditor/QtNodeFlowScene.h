#pragma once

#include "nodes/QFlowScene"

#include "SceneGraph.h"

namespace Qt
{

using dyno::SceneGraph;

/// Scene holds connections and nodes.
class QtNodeFlowScene
  : public QtFlowScene
{
	Q_OBJECT
public:

	QtNodeFlowScene(std::shared_ptr<QtDataModelRegistry> registry,
			QObject * parent = Q_NULLPTR);

	QtNodeFlowScene(QObject * parent = Q_NULLPTR);

	~QtNodeFlowScene();

	void addNodeByString(std::string NodeName);

	void enableEditing();
	void disableEditing();

public Q_SLOTS:
	/**
	 * @brief create a QT-based view for the active scene graph.
	 */
	void showSceneGraph();

	/**
	 * @brief Update the view only for the active scene graph, the data model will be changed.
	 */
	void updateSceneGraph();

	void moveModulePosition(QtNode& n, const QPointF& newLocation);

	void addNodeToSceneGraph(QtNode& n);

	void deleteNodeToSceneGraph(QtNode& n);
private:
	SceneGraph* m_scene = nullptr;

	bool mEditingEnabled = true;
};

}
