#pragma once

#include "QtFlowScene.h"

#include "Framework/SceneGraph.h"

namespace QtNodes
{

using dyno::SceneGraph;

/// Scene holds connections and nodes.
class NODE_EDITOR_PUBLIC QtNodeFlowScene
  : public QtFlowScene
{
	Q_OBJECT
public:

	QtNodeFlowScene(std::shared_ptr<DataModelRegistry> registry,
			QObject * parent = Q_NULLPTR);

	QtNodeFlowScene(QObject * parent = Q_NULLPTR);

	~QtNodeFlowScene();


public Q_SLOTS:
	void showSceneGraph(SceneGraph* scn);
	void moveModulePosition(QtBlock& n, const QPointF& newLocation);
private:
	SceneGraph* m_scene = nullptr;
};

}
