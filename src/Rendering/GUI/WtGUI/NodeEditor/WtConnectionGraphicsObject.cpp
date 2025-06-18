#include "WtConnectionGraphicsObject.h"

#include "WtFlowScene.h"

static Wt::WPainterPath cubicPath(WtConnectionGeometry const& geom)
{
	Wt::WPointF const& source = geom.source();

	Wt::WPointF const& sink = geom.sink();

	auto c1c2 = geom.pointsC1C2();

	//cubic spline
	Wt::WPainterPath cubic(source);

	cubic.cubicTo(c1c2.first, c1c2.second, sink);

	return cubic;
}

// No PainterPathStroker
//Wt::WPainterPath WtConnectionPainter::getPainterStroke(WtConnectionGeometry const& geom)
//{
//	auto cubic = cubicPath(geom);
//
//	Wt::WPointF const& source = geom.source();
//
//	Wt::WPainterPath result(source);
//
//	unsigned segments = 20;
//
//	for (auto i = 0ul; i < segments; ++i)
//	{
//		double ratio = double(i + 1) / segments;
//		//result.lineTo(cubic.pointAtPercent(ratio));
//	}
//
//	//QPainterPathStroker stroker; stroker.setWidth(10.0);
//
//	//return stroker.createStroke(result);
//}

static void drawSketchLine(Wt::WPainter* painter, WtConnection const& connection)
{
	WtConnectionState const& state = connection.connectionState();

	if (state.requiresPort())
	{
		auto const& connectionStyle = WtStyleCollection::connectionStyle();

		Wt::WPen p;
		p.setWidth(connectionStyle.constructionLineWidth());
		p.setColor(connectionStyle.constructionColor());
		p.setStyle(Wt::PenStyle::DashLine);

		painter->setPen(p);
		painter->setBrush(Wt::BrushStyle::None);

		WtConnectionGeometry const& geom = connection.connectionGeometry();
		auto cubic = cubicPath(geom);

		painter->drawPath(cubic);
	}
}

static void drawHoveredOrSelected(Wt::WPainter* painter, WtConnection const& connection)
{
	WtConnectionGeometry const& geom = connection.connectionGeometry();

	bool const hovered = geom.hovered();

	auto const& graphicsObject = connection.getConnectionGraphicsObject();

	bool const selected = graphicsObject.isSelected();

	// drawn as a fat background
	if (hovered || selected)
	{
		Wt::WPen p;

		auto const& connectionStyle = WtStyleCollection::connectionStyle();

		double const lineWidth = connectionStyle.lineWidth();

		p.setWidth(2 * lineWidth);
		p.setColor(Wt::WColor(Wt::StandardColor::Red));
		p.setColor(selected ? connectionStyle.selectedHaloColor() : connectionStyle.hoveredColor());

		painter->setPen(p);
		painter->setBrush(Wt::BrushStyle::None);

		auto cubic = cubicPath(geom);
		painter->drawPath(cubic);
	}
}

static void drawNormalLine(Wt::WPainter* painter, WtConnection const& connection)
{
	WtConnectionState const& state = connection.connectionState();

	if (state.requiresPort())
	{
		return;
	}

	// color
	auto const& connectionStyle = WtStyleCollection::connectionStyle();

	Wt::WColor normalColorOut = connectionStyle.normalColor();
	Wt::WColor normalColorIn = connectionStyle.normalColor();
	Wt::WColor selectedColor = connectionStyle.selectedColor();

	bool gradientColor = false;

	if (connectionStyle.useDataDefinedColors())
	{
		auto dataTypeOut = connection.dataType(PortType::Out);
		auto dataTypeIn = connection.dataType(PortType::In);

		gradientColor = (dataTypeOut.id != dataTypeIn.id);
		normalColorOut = connectionStyle.normalColor(dataTypeOut.id);
		normalColorIn = connectionStyle.normalColor(dataTypeIn.id);
		// no darker()
		//selectedColor = normalColorOut.darker(200);
	}

	//geometry
	WtConnectionGeometry const& geom = connection.connectionGeometry();

	double const lineWidth = connectionStyle.lineWidth();

	Wt::WPen p;
	p.setWidth(lineWidth);

	auto const& graphicsObject = connection.getConnectionGraphicsObject();
	bool const selected = graphicsObject.isSelected();

	// bug
	auto cubic = cubicPath(geom);

	if (gradientColor)
	{
		painter->setBrush(Wt::BrushStyle::None);

		Wt::WColor c = normalColorOut;
		//if (selected)
		// no darker()
		//c = c.darker(200);
		p.setColor(c);
		painter->setPen(p);

		unsigned int const segments = 60;

		for (unsigned int i = 0ul; i < segments; ++i)
		{
			double ratioPrev = double(i) / segments;
			double ratio = double(i + 1) / segments;

			if (i == segments / 2)
			{
				Wt::WColor c = normalColorIn;
				/*if (selected)
					c = c.darker(200);*/
				p.setColor(Wt::WColor(Wt::StandardColor::Red));
				painter->setPen(p);
			}
			//painter->drawLine(cubic.pointAtPercent(ratioPrev), cubic.pointAtPercent(ratio));
			//painter->drawLine(Wt::WPointF(10, 10), Wt::WPointF(100, 100));
			//painter->drawPath(cubic);
		}

		{
			//QIcon icon(":convert.png");

			//QPixmap pixmap = icon.pixmap(QSize(22, 22));
			//painter->drawPixmap(cubic.pointAtPercent(0.50) - QPoint(pixmap.width() / 2,
			//	pixmap.height() / 2),
			//	pixmap);
		}
	}
	else
	{
		p.setColor(normalColorOut);

		if (selected)
		{
			//p.setColor(selectedColor);
			p.setColor(Wt::WColor(Wt::StandardColor::Yellow));
		}



		painter->setPen(p);
		painter->setBrush(Wt::BrushStyle::None);
		painter->drawPath(cubic);
	}
}

void WtConnectionPainter::paint(
	Wt::WPainter* painter,
	WtConnection const& connection)
{
	drawHoveredOrSelected(painter, connection);

	drawSketchLine(painter, connection);

	drawNormalLine(painter, connection);

	// draw end points
	WtConnectionGeometry const& geom = connection.connectionGeometry();

	Wt::WPointF const& source = geom.source();
	Wt::WPointF const& sink = geom.sink();

	auto const& connectionStyle = WtStyleCollection::connectionStyle();

	double const pointDiameter = connectionStyle.pointDiameter();

	painter->setPen(connectionStyle.constructionColor());
	painter->setBrush(connectionStyle.constructionColor());
	double const pointRadius = pointDiameter / 4.0;
	//painter->drawEllipse(source, pointRadius, pointRadius);
	//painter->drawEllipse(sink, pointRadius, pointRadius);
}

WtConnectionGraphicsObject::WtConnectionGraphicsObject(WtFlowScene& scene, WtConnection& connection, Wt::WPainter* painter)
	: _scene(scene)
	, _connection(connection)
	, _painter(painter)
{
	//_scene.addItem(this);
	//setFlag(QGraphicsItem::ItemIsMovable, true);
	//setFlag(QGraphicsItem::ItemIsFocusable, true);
	//setFlag(QGraphicsItem::ItemIsSelectable, true);

	//setAcceptHoverEvents(true);

	// addGraphicsEffect();

	//setZValue(-1.0);

	//paint(_painter);
}

WtConnectionGraphicsObject::~WtConnectionGraphicsObject()
{
	//_scene.removeItem(this);
}

WtConnection& WtConnectionGraphicsObject::connection()
{
	return _connection;
}

Wt::WRectF WtConnectionGraphicsObject::boundingRect() const
{
	return _connection.connectionGeometry().boundingRect();
}

// No getPainterStroke()
//Wt::WPainterPath WtConnectionGraphicsObject::shape() const
//{
//	auto const& geom = _connection.connectionGeometry();
//
//	return WtConnectionPainter::getPainterStroke(geom);
//}

void WtConnectionGraphicsObject::setGeometryChanged()
{
	// QtItem
	//prepareGeometryChange();
}

void WtConnectionGraphicsObject::move()
{
	for (PortType portType : { PortType::In, PortType::Out })
	{
		if (auto node = _connection.getNode(portType))
		{
			auto const& nodeGraphics = node->nodeGraphicsObject();

			auto const& nodeGeom = node->nodeGeometry();

			Wt::WPointF origin = node->nodeGraphicsObject().getPos();

			Wt::WPointF scenePos = nodeGeom.portScenePosition(
				_connection.getPortIndex(portType),
				portType,
				nodeGraphics.sceneTransform());

			//Wt::WTransform sceneTransform = this->sceneTransform();
			Wt::WTransform sceneTransform(1, 0, 0, 1, 0, 0);

			Wt::WPointF connectionPos = sceneTransform.inverted().map(scenePos);

			Wt::WPointF result = Wt::WPointF(connectionPos.x() - origin.x(), connectionPos.y() - origin.y());

			_connection.connectionGeometry().setEndPoint(portType, result);

			//_connection.getConnectionGraphicsObject().setGeometryChanged();
			//_connection.getConnectionGraphicsObject().update();

		}
	}
	paint(_painter);
}

void WtConnectionGraphicsObject::lock(bool locked)
{
	/*setFlag(QGraphicsItem::ItemIsMovable, !locked);
	setFlag(QGraphicsItem::ItemIsFocusable, !locked);
	setFlag(QGraphicsItem::ItemIsSelectable, !locked);*/
}

void WtConnectionGraphicsObject::paint(Wt::WPainter* painter)
{
	//painter->setClipRect(option->exposedRect);

	WtConnectionPainter::paint(painter, _connection);
}

void WtConnectionGraphicsObject::addGraphicsEffect()
{
	//auto effect = new QGraphicsBlurEffect;

	//effect->setBlurRadius(5);
	//setGraphicsEffect(effect);

	//auto effect = new QGraphicsDropShadowEffect;
	//auto effect = new ConnectionBlurEffect(this);
	//effect->setOffset(4, 4);
	//effect->setColor(QColor(Qt::gray).darker(800));
}