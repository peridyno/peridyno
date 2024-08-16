#include "WtNodeGraphicsObject.h"

WtNodePainter::WtNodePainter() {}

WtNodePainter::~WtNodePainter() {}

void WtNodePainter::paint(Wt::WPainter* painter, WtNode& node, WtFlowScene const& scene)
{
	WtNodeGeometry const& geom = node.nodeGeometry();

	WtNodeState const& state = node.nodeState();

	WtNodeGraphicsObject const& graphicsObject = node.nodeGraphicsObject();

	WtNodeDataModel const* model = node.nodeDataModel();

	drawNodeRect(painter, geom, model, graphicsObject);
}

void WtNodePainter::drawNodeRect(
	Wt::WPainter* painter,
	WtNodeGeometry const& geom,
	WtNodeDataModel const* model,
	WtNodeGraphicsObject const& graphicsObject)
{
	WtNodeStyle const& nodeStyle = model->nodeStyle();
	//auto color = graphicsObject.isSelected() ? nodeStyle.SelectedBoundaryColor : nodeStyle.NormalBoundaryColor;

	auto color = nodeStyle.NormalBoundaryColor;
	if (geom.hovered())
	{
		Wt::WPen p(color);
		p.setWidth(nodeStyle.HoveredPenWidth);
		painter->setPen(p);
	}
	else
	{
		Wt::WPen p(color);
		p.setWidth(nodeStyle.PenWidth);
		painter->setPen(p);
	}

	float diam = nodeStyle.ConnectionPointDiameter;
	double const radius = 6.0;

	Wt::WRectF boundary = model->captionVisible() ? Wt::WRectF(-diam, -diam, 2.0 * diam + geom.width(), 2.0 * diam + geom.height())
		: Wt::WRectF(-diam, 0.0f, 2.0 * diam + geom.width(), diam + geom.height());

	//gradient

	if (model->captionVisible())
	{
		painter->drawRect(boundary);
	}
	else
	{
		painter->drawRect(boundary);
	}
}

void drawHotKeys(
	Wt::WPainter* painter,
	WtNodeGeometry const& geom,
	WtNodeDataModel const* model,
	WtNodeGraphicsObject const& graphicsObject)
{
	WtNodeStyle const& nodeStyle = model->nodeStyle();

	//auto color = graphicsObject.isSelected() ? nodeStyle.SelectedBoundaryColor : nodeStyle.NormalBoundaryColor;

	const Wt::WPen& pen = painter->pen();

	if (model->captionVisible() && model->hotkeyEnabled())
	{
		unsigned int captionHeight = geom.captionHeight();
		unsigned int keyWidth = geom.hotkeyWidth();
		unsigned int keyShift = geom.hotkeyIncline();
		unsigned int keyOffset = geom.hotkeyOffset();

		float diam = nodeStyle.ConnectionPointDiameter;

		//Wt different
		Wt::WRectF key0(geom.width() + diam - 20, -diam, 20, diam + captionHeight);

		double const radius = 6.0;

		//Wt different
		painter->setPen(Wt::WPen());
	}
}

void WtNodePainter::drawHotKeys(
	Wt::WPainter* painter,
	WtNodeGeometry const& geom,
	WtNodeDataModel const* model,
	WtNodeGraphicsObject const& graphicsObject)
{
	WtNodeStyle const& nodeStyle = model->nodeStyle();

	//auto color = graphicsObject.isSelected() ? nodeStyle.SelectedBoundaryColor : nodeStyle.NormalBoundaryColor;
}

//WtNodeGraphicsObject

WtNodeGraphicsObject::WtNodeGraphicsObject(WtFlowScene& scene, WtNode& node, Wt::WPainter* painter)
	: _scene(scene)
	, _node(node)
	, _painter(painter)
	, _locked(false)
{

	paint(_painter);
}

WtNodeGraphicsObject::~WtNodeGraphicsObject() {}

WtNode& WtNodeGraphicsObject::node()
{
	return _node;
}


WtNode const& WtNodeGraphicsObject::node() const
{
	return _node;
}

void WtNodeGraphicsObject::embedQWidget()
{

}

Wt::WRectF WtNodeGraphicsObject::boundingRect() const
{
	return _node.nodeGeometry().boundingRect();
}

void WtNodeGraphicsObject::setGeometryChanged()
{
	//prepareGeometryChange();
}

void WtNodeGraphicsObject::moveConnections() const
{
	WtNodeState const& nodeState = _node.nodeState();

	//for (PortType portType : {PortType::In, PortType::Out})
	//{
	//	auto const& connectionEntries = nodeState.getEntries(portType);

	//	for (auto const& connections : connectionEntries)
	//	{
	//		for (auto& con : connections)
	//			con.second->getConnectionGraphicsObject().move();
	//	}
	//}
}

void WtNodeGraphicsObject::lock(bool locked)
{
	_locked = locked;

	//setFlag(QGraphicsItem::ItemIsMovable, !locked);
	//setFlag(QGraphicsItem::ItemIsFocusable, !locked);
	//setFlag(QGraphicsItem::ItemIsSelectable, !locked);
}

void WtNodeGraphicsObject::paint(Wt::WPainter* painter)
{
	WtNodePainter::paint(painter, _node, _scene);
}

//bool WtNodeGraphicsObject::isSelected()
//{
//	return _selected;
//}
//
//void WtNodeGraphicsObject::setSelected(bool selected)
//{
//	_selected = selected;
//}