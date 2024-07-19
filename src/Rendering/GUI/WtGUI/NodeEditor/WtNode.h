#pragma once

#include <Wt/WPointF.h>
#include <Wt/WFontMetrics.h>
#include <Wt/WRectF.h>
#include <Wt/WFont.h>
#include <Wt/WTransform.h>

#include "WtNodeGraphicsObject.h"
#include "WtNodeDataModel.h"
#include "WtNodeDate.hpp"
#include "WtNodeStyle.h"
#include <memory>

class WtNodeGraphicsObject;
class WtNodeDataModel;
class WtStyleCollection;

class NodeGeometry
{
public:
	NodeGeometry(std::unique_ptr<WtNodeDataModel> const& dataModel);
	~NodeGeometry();

public:
	unsigned int height() const { return _height; };

	void setHeight(unsigned int h) { _height = h; }

	unsigned int width() const { return _width; }

	void setWidth(unsigned int w) { _width = w; }

	unsigned int entryHeight() const { return _entryHeight; }

	void setEntryHeight(unsigned int h) { _entryHeight = h; }

	unsigned int entryWidth() const { return _entryWidth; }

	void setEntryWidth(unsigned int w) { _entryWidth = w; }

	unsigned int hotkeyWidth() const { return _hotkeyWidth; }

	void setHotkeyWidth(unsigned int w) { _hotkeyWidth = w; }

	unsigned int hotkeyIncline() const { return _hotkeyIncline; }

	void setHotkeyIncline(unsigned int inc) { _hotkeyIncline = inc; }

	unsigned int hotkeyOffset() const { return _hotkeyOffset; }

	void setHotkeyOffset(unsigned int offset) { _hotkeyOffset = offset; }

	unsigned int spacing() const { return _spacing; }

	unsigned int captionHeightSpacing() const { return _captionHeightSpacing; }

	void setSpacing(unsigned int s) { _spacing = s; }

	bool hovered() const { return _hovered; }

	void setHovered(unsigned int h) { _hovered = h; }

	unsigned int nSources() const;

	unsigned int nSinks() const;

	Wt::WPointF const& draggingPos() const
	{
		return _draggingPos;
	}

	void setDraggingPosition(Wt::WPointF const& pos)
	{
		_draggingPos = pos;
	}

	//QDialog* TipsWidget;
	//QDockWidget* TipsWidget_;
	//QDialog* PortTipsWidget;

	PortType hoverport_type;
	PortIndex hoverport_id;

	bool* isPortTipsShow;
	bool* isNodeTipsShow;

	void ShowTips()const;
	void HideTips()const;

	bool getPortTipsState()const;
	void ShowPortTips()const;
	void HidePortTips()const;

public:
	Wt::WRectF entryBoundingRect() const;

	Wt::WRectF boundingRect() const;

	void recalculateSize() const;

	void recalculateSize(Wt::WFont const& font) const;

	Wt::WPointF portScenePosition(PortType index, PortType portType, Wt::WTransform const& t = Wt::WTransform()) const;

	PortIndex checkHitScenePoint(PortType portType, Wt::WPointF point, Wt::WTransform const& t = Wt::WTransform()) const;

	PortIndex hoverHitScenePoint(PortType portType, Wt::WPointF point, Wt::WTransform const& t = Wt::WTransform()) const;

	PortIndex hoverHitPortArea(PortType portType,
		Wt::WPointF const scenePoint,
		Wt::WTransform const& sceneTransform,
		NodeGeometry const& geom,
		WtNodeDataModel const* mode) const;

	PortIndex findHitPort(PortType portType,
		Wt::WPointF const scenePoint,
		Wt::WTransform const& sceneTransform,
		NodeGeometry const& geom,
		WtNodeDataModel const* model) const;

	bool checkHitHotKey0(Wt::WPointF point, Wt::WTransform const& t = Wt::WTransform()) const;

	bool checkHitHotKey1(Wt::WPointF point, Wt::WTransform const& t = Wt::WTransform()) const;

	//QRect resizeRect() const;

	Wt::WPointF widgetPosition() const;

	int equivalentWidgetHeight() const;

	unsigned int validationHeight() const;

	unsigned int validationWidth() const;

	//static Wt::WPointF calculateNodePositionBetweenNodePorts(
	//	PortIndex targetPortIndex, PortType targetPort, WtNode* targetNode,
	//	PortIndex sourcePortIndex, PortType sourcePort, WtNode* sourceNode,
	//	WtNode& newNode);

	unsigned int captionHeight() const;

	unsigned int captionWidth() const;

private:
	unsigned int portWidth(PortType portType) const;

private:
	mutable unsigned int _width;
	mutable unsigned int _height;
	unsigned int _entryWidth;
	mutable unsigned int _inputPortWidth;
	mutable unsigned int _outputPortWidth;
	mutable unsigned int _entryHeight;
	unsigned int _spacing;
	unsigned int _captionHeightSpacing;

	unsigned int _hotkeyWidth;
	unsigned int _hotkeyIncline;
	unsigned int _hotkeyOffset;

	bool _hovered;

	unsigned int _nSources;
	unsigned int _nSinks;

	Wt::WPointF _draggingPos;

	std::unique_ptr<WtNodeDataModel> const& _dataModel;

	//mutable Wt::WFontMetrics _fontMetrics;
	//mutable Wt::WFontMetrics _boldFontMetrics;
};

class NodeState
{
public:
	enum ReactToConnectionState
	{
		REACTING,
		NOT_REACTING
	};

public:
	NodeState(std::unique_ptr<WtNodeDataModel> const& model);

};

class WtNode
{
public:
	WtNode(std::unique_ptr<WtNodeDataModel>&& dataModel);

	virtual ~WtNode();

public:
	//QJsonObject save() const override;

	//void restore(QJsonObject const& json) override;

public:
	//QUuid id() const;

	void reactToPossibleConnection(PortType,
		NodeDataType const&,
		Wt::WPointF const& scenePoint);

	void resetReactionToConnection();

public:

	WtNodeGraphicsObject const& nodeGraphicsObject() const;

	WtNodeGraphicsObject& nodeGraphicsObject();

	void setGraphicsObject(std::unique_ptr<WtNodeGraphicsObject>&& graphics);

	NodeGeometry& nodeGeometry();

	NodeGeometry const& nodeGeometry() const;

	NodeState const& nodeState() const;

	NodeState& nodeState();

	WtNodeDataModel* nodeDataModel() const;

public:
	/// Propagates incoming data to the underlying model.
	void propagateData(std::shared_ptr<WtNodeData> nodeData, PortIndex inPortIndex) const;

	/// Fetches data from model's OUT #index port
	/// and propagates it to the connection
	void onDataUpdated(PortIndex index);

	/// update the graphic part if the size of the embeddedwidget changes
	void onNodeSizeUpdated();

private:
	//QUuid _uid;

	std::unique_ptr<WtNodeDataModel> _nodeDataModel;

	NodeState _nodeState;

	NodeGeometry _nodeGeometry;

	//std::unique_ptr<WtNodeGraphicsObject> _nodeGraphicsObject;

};