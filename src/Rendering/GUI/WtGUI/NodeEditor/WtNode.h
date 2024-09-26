#pragma once

#include <Wt/WPointF.h>
#include <Wt/WFontMetrics.h>
#include <Wt/WRectF.h>
#include <Wt/WFont.h>
#include <Wt/WTransform.h>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_hash.hpp>

#include "WtNodeGraphicsObject.h"
#include "WtNodeDataModel.h"
#include "WtNodeData.hpp"
#include "WtNodeStyle.h"
#include <memory>
#include <unordered_map>

class WtNodeGraphicsObject;
class WtNodeDataModel;
class WtStyleCollection;

class WtNodeGeometry
{
public:
	WtNodeGeometry(std::unique_ptr<WtNodeDataModel> const& dataModel, Wt::WPaintDevice* paintDevice);
	~WtNodeGeometry();

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

	//void recalculateSize(Wt::WFont const& font, Wt::WFontMetrics fontMetrics) const;

	Wt::WPointF portScenePosition(PortIndex index, PortType portType, Wt::WTransform const& t = Wt::WTransform()) const;

	PortIndex checkHitScenePoint(PortType portType, Wt::WPointF point, Wt::WTransform const& t = Wt::WTransform()) const;

	PortIndex hoverHitScenePoint(PortType portType, Wt::WPointF point, Wt::WTransform const& t = Wt::WTransform()) const;

	PortIndex hoverHitPortArea(PortType portType,
		Wt::WPointF const scenePoint,
		Wt::WTransform const& sceneTransform,
		WtNodeGeometry const& geom,
		WtNodeDataModel const* mode,
		Wt::WFontMetrics const& metrics) const;

	PortIndex findHitPort(PortType portType,
		Wt::WPointF const scenePoint,
		Wt::WTransform const& sceneTransform,
		WtNodeGeometry const& geom,
		WtNodeDataModel const* model) const;

	bool checkHitHotKey0(Wt::WPointF point, Wt::WTransform const& t = Wt::WTransform()) const;

	bool checkHitHotKey1(Wt::WPointF point, Wt::WTransform const& t = Wt::WTransform()) const;

	Wt::WRectF resizeRect() const;

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

	mutable Wt::WFontMetrics _fontMetrics;
	//mutable Wt::WFontMetrics _boldFontMetrics;
};

class WtNodeState
{
public:
	enum ReactToConnectionState
	{
		REACTING,
		NOT_REACTING
	};

public:
	WtNodeState(std::unique_ptr<WtNodeDataModel> const& model);

public:
	using ConnectionPtrSet = std::unordered_map<boost::uuids::uuid, WtConnection*>;

	/// Returns vector of connections ID.
	/// Some of them can be empty (null)
	std::vector<ConnectionPtrSet> const& getEntries(PortType) const;

	std::vector<ConnectionPtrSet>& getEntries(PortType);

	ConnectionPtrSet connections(PortType portType, PortIndex portIndex) const;

	void setConnection(
		PortType portType,
		PortIndex portIndex,
		WtConnection& connection);

	void eraseConnection(
		PortType portType,
		PortIndex portIndex,
		boost::uuids::uuid id);

	ReactToConnectionState reaction() const;

	PortType reactingPortType() const;

	NodeDataType reactingDataType() const;

	void setReaction(
		ReactToConnectionState reaction,
		PortType reactingPortType = PortType::None,
		NodeDataType reactingDataType =
		NodeDataType());

	bool isReacting() const;

	void setResizing(bool resizing);

	bool resizing() const;

private:
	std::vector<ConnectionPtrSet> _inConnections;
	std::vector<ConnectionPtrSet> _outConnections;

	ReactToConnectionState _reaction;
	PortType     _reactingPortType;
	NodeDataType _reactingDataType;

	bool _resizing;
};

class WtNode
{
public:
	WtNode(std::unique_ptr<WtNodeDataModel>&& dataModel, Wt::WPaintDevice* paintDevice);

	virtual ~WtNode();

public:
	//QJsonObject save() const override;

	//void restore(QJsonObject const& json) override;

public:
	boost::uuids::uuid id() const;

	void reactToPossibleConnection(PortType,
		NodeDataType const&,
		Wt::WPointF const& scenePoint);

	void resetReactionToConnection();

public:

	WtNodeGraphicsObject const& nodeGraphicsObject() const;

	WtNodeGraphicsObject& nodeGraphicsObject();

	void setGraphicsObject(std::unique_ptr<WtNodeGraphicsObject>&& graphics);

	WtNodeGeometry& nodeGeometry();

	WtNodeGeometry const& nodeGeometry() const;

	WtNodeState const& nodeState() const;

	WtNodeState& nodeState();

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
	boost::uuids::uuid _uid;

	std::unique_ptr<WtNodeDataModel> _nodeDataModel;

	WtNodeState _nodeState;

	WtNodeGeometry _nodeGeometry;

	std::unique_ptr<WtNodeGraphicsObject> _nodeGraphicsObject;
};