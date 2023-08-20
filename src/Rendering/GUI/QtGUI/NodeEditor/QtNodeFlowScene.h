#pragma once

#include "nodes/QFlowScene"

#include "SceneGraph.h"
#include "FBase.h"

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

	void setDx(float dx) { mDx = dx; }
	void setDy(float dy) { mDy = dy; }

	float dx() { return mDx; }
	float dy() { return mDy; }

public Q_SLOTS:
	/**
	 * @brief create a QT-based view for the active scene graph.
	 */
	void createNodeGraphView();

	/**
	 * @brief Update the view only for the active scene graph, the data model will not be changed.
	 */
	void updateNodeGraphView();

	void fieldUpdated(dyno::FBase* field, int status);

	void addNode(QtNode& n);

	void deleteNode(QtNode& n);

	void moveNode(QtNode& n, const QPointF& newLocation);

	void createQtNode(std::shared_ptr<dyno::Node> node);

	void enableRendering(QtNode& n, bool checked);

	void enablePhysics(QtNode& n, bool checked);

	void showContextMenu(QtNode& n, const QPointF& pos);

	void showHelper(QtNode& n);

	/**
	 * Auto layout for the node graph
	 */
	void reorderAllNodes();

private:
	void showThisNodeOnly(QtNode& n);
	void showAllNodes();

	SceneGraph* m_scene = nullptr;

	bool mEditingEnabled = true;

	float mDx = 100.0f;
	float mDy = 50.0f;
};

}
