#pragma once

#include <QtGui/QPainter>

namespace QtNodes
{

class ConnectionGeometry;
class ConnectionState;
class QtConnection;

class ConnectionPainter
{
public:

  static
  void
  paint(QPainter* painter,
        QtConnection const& connection);

  static
  QPainterPath
  getPainterStroke(ConnectionGeometry const& geom);
};
}
