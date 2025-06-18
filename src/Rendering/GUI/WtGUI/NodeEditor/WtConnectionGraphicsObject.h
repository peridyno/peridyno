#pragma once

#include <Wt/WPainter.h>
#include <Wt/WPainterPath.h>
#include <Wt/WRectF.h>

#include "guid.hpp"

#include "WtConnection.h"

class WtNode;
class WtFlowScene;
class WtConnection;
class WtConnectionGeometry;
class WtConnectionState;

class WtConnectionPainter
{
public:

	static void paint(Wt::WPainter* painter, WtConnection const& connection);

	static Wt::WPainterPath getPainterStroke(WtConnectionGeometry const& geom);
};

/// Graphic Object for connection. Adds itself to scene
class WtConnectionGraphicsObject
{
public:

	WtConnectionGraphicsObject(WtFlowScene& scene, WtConnection& connection, Wt::WPainter* painter);

	virtual ~WtConnectionGraphicsObject();

	enum { Type = 65538 };
	int type() const { return Type; }

public:

	WtConnection& connection();

	Wt::WRectF boundingRect() const;

	Wt::WPainterPath shape() const;

	void setGeometryChanged();

	/// Updates the position of both ends
	void move();

	void lock(bool locked);

	void setPos(int x, int y)
	{
		_origin.setX(x);
		_origin.setY(y);
		_painter->translate(_origin);
		paint(_painter);
		_origin.setX(-x);
		_origin.setY(-y);
		_painter->translate(_origin);
	}

	void setPos(Wt::WPointF pos)
	{
		_origin = pos;
		_painter->translate(_origin);
		paint(_painter);
		_origin.setX(-pos.x());
		_origin.setY(-pos.y());
		_painter->translate(_origin);
	}

	bool isSelected() const
	{
		return false;
	}
protected:

	void paint(Wt::WPainter* painter);
	//
	//	void
	//		mousePressEvent(QGraphicsSceneMouseEvent* event) override;
	//
	//	void
	//		mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
	//
	//	void
	//		mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
	//
	//	void
	//		hoverEnterEvent(QGraphicsSceneHoverEvent* event) override;
	//
	//	void
	//		hoverLeaveEvent(QGraphicsSceneHoverEvent* event) override;

private:

	void addGraphicsEffect();

private:

	WtFlowScene& _scene;

	WtConnection& _connection;

	// use for move
	Wt::WPointF _origin = Wt::WPointF(0, 0);

	Wt::WPainter* _painter;
};