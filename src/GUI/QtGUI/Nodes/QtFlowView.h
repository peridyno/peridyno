#pragma once

#include <QtWidgets/QGraphicsView>

#include "Export.h"

namespace QtNodes
{

class QtFlowScene;

class NODE_EDITOR_PUBLIC QtFlowView
  : public QGraphicsView
{
  Q_OBJECT
public:

  QtFlowView(QWidget *parent = Q_NULLPTR);
  QtFlowView(QtFlowScene *scene, QWidget *parent = Q_NULLPTR);

  QtFlowView(const QtFlowView&) = delete;
  QtFlowView operator=(const QtFlowView&) = delete;

  QAction* clearSelectionAction() const;

  QAction* deleteSelectionAction() const;

  void setScene(QtFlowScene *scene);

public Q_SLOTS:

  void scaleUp();

  void scaleDown();

  void deleteSelectedNodes();

protected:

  void contextMenuEvent(QContextMenuEvent *event) override;

  void wheelEvent(QWheelEvent *event) override;

  void keyPressEvent(QKeyEvent *event) override;

  void keyReleaseEvent(QKeyEvent *event) override;

  void mousePressEvent(QMouseEvent *event) override;

  void mouseMoveEvent(QMouseEvent *event) override;

  void drawBackground(QPainter* painter, const QRectF& r) override;

  void showEvent(QShowEvent *event) override;

protected:

  QtFlowScene * scene();

private:

  QAction* _clearSelectionAction;
  QAction* _deleteSelectionAction;

  QPointF _clickPos;

  QtFlowScene* _scene;
};
}
