#pragma once

#include <Wt/WPainter.h>
#include <Wt/WPainterPath.h>
#include <Wt/WRectF.h>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

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

	WtConnectionGraphicsObject(WtFlowScene& scene, WtConnection& connection);

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

	//protected:
	//
	//	void paint(Wt::WPainter* painter,
	//		QStyleOptionGraphicsItem const* option,
	//		QWidget* widget = 0) override;
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
};