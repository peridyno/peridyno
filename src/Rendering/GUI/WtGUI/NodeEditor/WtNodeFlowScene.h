#pragma once

#include "WtFlowScene.h"
#include "WtNodeData.hpp"
#include "WtNodeWidget.h"

#include "SceneGraph.h"
#include "FBase.h"

using dyno::SceneGraph;

class WtNodeFlowScene : public WtFlowScene
{
public:
	//WtNodeFlowScene(std::shared_ptr<WtDataModelRegistry> registry, Wt::WPainter* painter);
	WtNodeFlowScene(Wt::WPainter* painter);
	~WtNodeFlowScene();

	void addNodeByString(std::string NodeName);

	void enableEditing();
	void disableEditing();

	void setDx(float dx) { mDx = dx; }
	void setDy(float dy) { mDy = dy; }

	float dx() { return mDx; }
	float dy() { return mDy; }

public:
	/**
 * @brief create a QT-based view for the active scene graph.
 */
	void createNodeGraphView();

	/**
	 * @brief Update the view only for the active scene graph, the data model will not be changed.
	 */
	void updateNodeGraphView();

	void fieldUpdated(dyno::FBase* field, int status);

	void addNode(WtNode& n);

	void deleteNode(WtNode& n);

	void moveNode(WtNode& n, const Wt::WPointF& newLocation);

	void createWtNode(std::shared_ptr<dyno::Node> node);

	void enableRendering(WtNode& n, bool checked);

	void enablePhysics(WtNode& n, bool checked);

	void showContextMenu(WtNode& n, const Wt::WPointF& pos);

	void showHelper(WtNode& n);

	/**
	 * Auto layout for the node graph
	 */
	void reorderAllNodes();

private:
	void showThisNodeOnly(WtNode& n);
	void showAllNodes();

	void activateThisNodeOnly(WtNode& n);
	void activateAllNodes();

	void autoSyncAllNodes(bool autoSync);

	void autoSyncAllDescendants(WtNode& n, bool autoSync);
private:
	SceneGraph* m_scene = nullptr;

	bool mEditingEnabled = true;

	float mDx = 100.0f;
	float mDy = 50.0f;
};