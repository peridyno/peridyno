#include "QtConnectionBlurEffect.h"

#include "QtConnectionGraphicsObject.h"
#include "ConnectionPainter.h"

using QtNodes::QtConnectionBlurEffect;
using QtNodes::QtConnectionGraphicsObject;

QtConnectionBlurEffect::
QtConnectionBlurEffect(QtConnectionGraphicsObject*)
{
  //
}


void
QtConnectionBlurEffect::
draw(QPainter* painter)
{
  QGraphicsBlurEffect::draw(painter);

  //ConnectionPainter::paint(painter,
  //_object->connectionGeometry(),
  //_object->connectionState());

  //_item->paint(painter, nullptr, nullptr);
}
