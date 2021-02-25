#pragma once

#include <QtGui/QPainter>

namespace QtNodes
{

class QtBlock;
class BlockState;
class BlockGeometry;
class QtBlockGraphicsObject;
class QtBlockDataModel;
class FlowItemEntry;
class QtFlowScene;

class BlockPainter
{
public:

  BlockPainter();

public:

  static
  void
  paint(QPainter* painter,
        QtBlock& node,
        QtFlowScene const& scene);

  static
  void
  drawNodeRect(QPainter* painter,
               BlockGeometry const& geom,
               QtBlockDataModel const* model,
               QtBlockGraphicsObject const & graphicsObject);

  static
  void
  drawModelName(QPainter* painter,
                BlockGeometry const& geom,
                BlockState const& state,
                QtBlockDataModel const * model);

  static
  void
  drawEntryLabels(QPainter* painter,
                  BlockGeometry const& geom,
                  BlockState const& state,
                  QtBlockDataModel const * model);

  static
  void
  drawConnectionPoints(QPainter* painter,
                       BlockGeometry const& geom,
                       BlockState const& state,
                       QtBlockDataModel const * model,
                       QtFlowScene const & scene);

  static
  void
  drawFilledConnectionPoints(QPainter* painter,
                             BlockGeometry const& geom,
                             BlockState const& state,
                             QtBlockDataModel const * model);

  static
  void
  drawResizeRect(QPainter* painter,
                 BlockGeometry const& geom,
                 QtBlockDataModel const * model);

  static
  void
  drawValidationRect(QPainter * painter,
                     BlockGeometry const & geom,
                     QtBlockDataModel const * model,
                     QtBlockGraphicsObject const & graphicsObject);
};
}
