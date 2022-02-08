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


public Q_SLOTS:
	void showSceneGraph(SceneGraph* scn);
	void moveModulePosition(QtNode& n, const QPointF& newLocation);

	void addNodeToSceneGraph(QtNode& n);

	void deleteNodeToSceneGraph(QtNode& n);
private:
	SceneGraph* m_scene = nullptr;
};

}
