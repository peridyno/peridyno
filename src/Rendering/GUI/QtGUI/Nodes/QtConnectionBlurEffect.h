#include <QtWidgets/QGraphicsBlurEffect>

#include <QtWidgets/QGraphicsItem>

namespace QtNodes
{

class QtConnectionGraphicsObject;

class QtConnectionBlurEffect : public QGraphicsBlurEffect
{

public:

  QtConnectionBlurEffect(QtConnectionGraphicsObject* item);

  void
  draw(QPainter* painter) override;

private:
};
}
