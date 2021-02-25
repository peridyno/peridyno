#pragma once

#include <QPainter>

#include "BlockGeometry.h"
#include "QtBlockDataModel.h"
#include "Export.h"

namespace QtNodes {

/// Class to allow for custom painting
class NODE_EDITOR_PUBLIC BlockPainterDelegate
{

public:

  virtual
  ~BlockPainterDelegate() = default;

  virtual void
  paint(QPainter* painter,
        BlockGeometry const& geom,
        QtBlockDataModel const * model) = 0;
};
}
