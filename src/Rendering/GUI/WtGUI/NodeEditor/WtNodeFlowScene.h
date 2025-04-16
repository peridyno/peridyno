#pragma once

#include "WtFlowScene.h"
#include "WtNodeData.hpp"
#include "WtNodeWidget.h"

#include "SceneGraphFactory.h"
#include "SceneGraph.h"
#include "FBase.h"
#include "Action.h"
#include "DirectedAcyclicGraph.h"
#include "AutoLayoutDAG.h"

using dyno::SceneGraph;

class WtNodeFlowScene : public WtFlowScene
{
public:
	WtNodeFlowScene(std::shared_ptr<WtDataModelRegistry> registry, Wt::WPainter* painter);
	WtNodeFlowScene(Wt::WPainter* painter, std::shared_ptr<dyno::SceneGraph> scene, int selectType, int selectNum);
	~WtNodeFlowScene();

	void addNode(WtNode& n);

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

	void createWtNode(std::shared_ptr<dyno::Node> node);

	void enableRendering(WtNode& n, bool checked);

	void enablePhysics(WtNode& n, bool checked);

	void showContextMenu(WtNode& n, const Wt::WPointF& pos);

	void showHelper(WtNode& n);

	/**
	 * Auto layout for the node graph
	 */
	void reorderAllNodes();

	std::map<dyno::ObjectId, WtNode*> getNodeMap();

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

	Wt::WPainter* _painter;

	float mDx = 100.0f;
	float mDy = 50.0f;

	std::shared_ptr<dyno::SceneGraph> mScene = nullptr;

	std::map<dyno::ObjectId, WtNode*> OutNodeMap;

	int _selectType;
	int _selectNum;

	bool _isSelectedPoint;
	Wt::WPointF _mousePoint = Wt::WPointF(0, 0);
};