#pragma once

#include <Wt/WPaintDevice.h>
#include <Wt/WPainter.h>
#include <Wt/WColor.h>
#include <Wt/WPen.h>
#include <Wt/WGradient.h>

#include "WtNode.h"
#include "WtNodeStyle.h"
#include "WtFlowScene.h"

class WtNode;
class WtNodeState;
class WtNodeGeometry;
class WtNodeGraphicsObject;
class WtNodeDataModel;
class WtFlowScene;

class WtNodePainter
{
public:

	WtNodePainter();

	~WtNodePainter();

public:

	static void paint(Wt::WPainter* painter, WtNode& node);

	static void drawNodeRect(
		Wt::WPainter* painter,
		WtNodeGeometry const& geom,
		WtNodeDataModel const* model,
		WtNodeGraphicsObject const& graphicsObject);

	static void drawHotKeys(
		Wt::WPainter* painter,
		WtNodeGeometry const& geom,
		WtNodeDataModel const* model,
		WtNodeGraphicsObject const& graphicsObject);

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
	WtNodeGraphicsObject(WtFlowScene& scene, WtNode& node);
	virtual ~WtNodeGraphicsObject();
	WtNode& node();
	WtNode const& node() const;
	Wt::WRectF boundingRect() const;
	void setGeometryChanged();

	/// Visits all attached connections and corrects
	/// their corresponding end points.
	void moveConnections() const;

	enum { Type = 65537 };

	int	type() const { return Type; }

	void lock(bool locked);

	bool isHotKey0Checked() const { return _hotKey0Checked; }

	void setHotKey0Checked(bool checked) { _hotKey0Checked = checked; }

	bool hotKey0Hovered() const { return _hotKey0Hovered; }

	void setHotKey0Hovered(bool h) { _hotKey0Hovered = h; }

	bool isHotKey1Checked() const { return _hotKey1Checked; }

	void setHotKey1Checked(bool checked) { _hotKey1Checked = checked; }

	bool hotKey1Hovered() const { return _hotKey1Hovered; }

	void setHotKey1Hovered(bool h) { _hotKey1Hovered = h; }

	void isSelect();

private:
	WtFlowScene& _scene;

	WtNode& _node;

	Wt::WPainter* _defaultPainter;

	bool _locked;

	bool _hotKey0Checked = true;

	bool _hotKey0Hovered = false;

	bool _hotKey1Hovered = false;

	bool _hotKey1Checked = true;

	bool _hotKey2Hovered = false;

	bool _hotKey2Checked = false;

	int _pressCounter = 0;

	int HelpTimerID = -1;
	int PortTimerID = -1;

	// either nullptr or owned by parent QGraphicsItem
	//QGraphicsProxyWidget* _proxyWidget;
};