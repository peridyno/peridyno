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

	drawHotKeys(painter, geom, model, graphicsObject);

	drawConnectionPoints(painter, geom, state, model, scene);

	drawModelName(painter, geom, state, model);

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
	WtNodeGraphicsObject const& graphicsObject
)
{
	WtNodeStyle const& nodeStyle = model->nodeStyle();

	//auto color = graphicsObject.isSelected() ? nodeStyle.SelectedBoundaryColor : nodeStyle.NormalBoundaryColor;

	auto color = nodeStyle.SelectedBoundaryColor;

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
		Wt::WPen pen = Wt::WPen(color);
		pen.setWidth(nodeStyle.HoveredPenWidth);
		painter->setPen(pen);

		if (graphicsObject.hotKey0Hovered())
		{
			pen.setWidth(nodeStyle.HoveredPenWidth);
			painter->setPen(pen);
		}
		else
		{
			pen.setWidth(nodeStyle.PenWidth);
			painter->setPen(pen);
		}

		if (graphicsObject.isHotKey0Checked())
		{
			painter->setBrush(nodeStyle.GradientColor0);
		}
		else
		{
			painter->setBrush(nodeStyle.HotKeyColor0);
		}

		Wt::WPointF points[4];
		points[0] = Wt::WPointF(geom.width() + diam - keyWidth - keyOffset, -diam);
		points[1] = Wt::WPointF(geom.width() + diam - keyOffset, -diam);
		points[2] = Wt::WPointF(geom.width() + diam - keyShift - keyOffset, captionHeight);
		points[3] = Wt::WPointF(geom.width() + diam - keyWidth - keyShift - keyOffset, captionHeight);

		painter->drawPolygon(points, 4);

		if (graphicsObject.hotKey1Hovered())
		{
			pen.setWidth(nodeStyle.HoveredPenWidth);
			painter->setPen(pen);
		}
		else
		{
			pen.setWidth(nodeStyle.PenWidth);
			painter->setPen(pen);
		}

		if (graphicsObject.isHotKey1Checked())
		{
			painter->setBrush(nodeStyle.GradientColor0);
		}
		else
		{
			painter->setBrush(nodeStyle.HotKeyColor1);
		}

		points[0] = Wt::WPointF(geom.width() + diam - keyWidth - keyOffset, -diam);
		points[1] = Wt::WPointF(geom.width() + diam - keyWidth - keyShift - keyOffset, captionHeight);
		points[2] = Wt::WPointF(geom.width() + diam - keyWidth - keyShift - keyWidth - keyOffset, captionHeight);
		points[3] = Wt::WPointF(geom.width() + diam - keyWidth - keyWidth - keyOffset, -diam);

		painter->drawPolygon(points, 4);
	}
}

void WtNodePainter::drawConnectionPoints(
	Wt::WPainter* painter,
	WtNodeGeometry const& geom,
	WtNodeState const& state,
	WtNodeDataModel const* model,
	WtFlowScene const& scene
)
{
	WtNodeStyle const& nodeStyle = model->nodeStyle();
	auto const& connectionStyle = WtStyleCollection::connectionStyle();

	float diameter = nodeStyle.ConnectionPointDiameter;
	auto reducedDiameter = diameter * 0.6;

	for (PortType portType : {PortType::Out, PortType::In})
	{
		//size_t n = state.getEntries(portType).size();

		int n = 10;
		for (unsigned int i = 0; i < n; i++)
		{
			Wt::WPointF p = geom.portScenePosition(i, portType);

			//TODO:Bug
			//auto const& dataType = model->dataType(portType, i);

			//bool canConnect = (state.getEntries(portType)[i].empty() ||
			//	(portType == PortType::Out &&
			//		model->portOutConnectionPolicy(i) == WtNodeDataModel::ConnectionPolicy::Many));

			//double r = 1.0;

			//if (state.isReacting() && canConnect && portType == state.reactingPortType())
			//{
			//	//auto diff = geom.draggingPos() - p;
			//	//double dist = std::sqrt();
			//	bool typeConvertable = false;
			//}
		}

	}
}

void WtNodePainter::drawModelName(
	Wt::WPainter* painter,
	WtNodeGeometry const& geom,
	WtNodeState const& state,
	WtNodeDataModel const* model
)
{
	WtNodeStyle const& nodeStyle = model->nodeStyle();

	if (!model->captionVisible())
		return;

	std::string const& name = model->caption();

	Wt::WFont f = painter->font();

	f.setWeight(Wt::FontWeight::Bold);

	Wt::WRectF position(10, 10, 10, 10);
	float diam = nodeStyle.ConnectionPointDiameter;
	Wt::WRectF boundary = model->captionVisible() ? Wt::WRectF(-diam, -diam, 2.0 * diam + geom.width(), 2.0 * diam + geom.height())
		: Wt::WRectF(-diam, 0.0f, 2.0 * diam + geom.width(), diam + geom.height());

	painter->setFont(f);
	//painter->setPen(nodeStyle.FontColor);
	painter->drawText(boundary, Wt::AlignmentFlag::Left, Wt::WString(name));

	f.setWeight(Wt::FontWeight::Normal);
	painter->setFont(f);
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