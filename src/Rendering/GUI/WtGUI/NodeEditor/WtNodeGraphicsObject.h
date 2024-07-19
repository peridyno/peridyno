#pragma once

#include <Wt/WPaintDevice.h>
#include <Wt/WPainter.h>
#include <Wt/WColor.h>

#include "WtNode.h"
#include "WtNodeStyle.h"

class WtNode;
class NodeState;
class NodeGeometry;
class WtNodeGraphicsObject;
class WtNodeDataModel;
class WtFlowScene;

class WtNodePainter
{
public:
	WtNodePainter();
	~WtNodePainter();

public:
	static void paint(Wt::WPainter* painter);
	static void drawNodeRect(Wt::WPainter* painter, NodeGeometry const& geom, WtNodeDataModel const* model, WtNodeGraphicsObject& graphicsObject);
	static void drawHotKeys(Wt::WPainter* painter);
	static void drawModelName(Wt::WPainter* painter);
	static void drawEntryLabels(Wt::WPainter* painter);
	static void drawConnectionPoints(Wt::WPainter* painter);
	static void drawFilledConnectionPoints(Wt::WPainter* painter);
	static void drawResizeRect(Wt::WPainter* painter);
	static void drawValidationRect(Wt::WPainter* painter);
};

class WtNodeGraphicsObject
{
public:
	WtNodeGraphicsObject();
	virtual ~WtNodeGraphicsObject();

public:
	bool isSelected();
	void setSelected(bool selected);

private:
	bool _selected = false;
};