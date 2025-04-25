#pragma once

#include <Wt/WEvent.h>
#include <Wt/WPaintedWidget.h>
#include <Wt/WPointF.h>
#include <Wt/WPainterPath.h>
#include <Wt/WPainter.h>

#include <SceneGraph.h>

class WtFlowWidget : public Wt::WPaintedWidget
{
public:
	WtFlowWidget(std::shared_ptr<dyno::SceneGraph> scene);
	virtual ~WtFlowWidget() = default;

	virtual void onMouseMove(const Wt::WMouseEvent& event) = 0;
	virtual void onMouseWentDown(const Wt::WMouseEvent& event) = 0;
	virtual void onMouseWentUp(const Wt::WMouseEvent& event) = 0;
	virtual void onKeyWentDown() = 0;

	void onMouseWheel(const Wt::WMouseEvent& event);
	void zoomIn();
	void zoomOut();

	void reorderNode();
	void updateAll();

	Wt::WPainterPath cubicPath(Wt::WPointF source, Wt::WPointF sink);
	std::pair<Wt::WPointF, Wt::WPointF> pointsC1C2(Wt::WPointF source, Wt::WPointF sink);
	void drawSketchLine(Wt::WPainter* painter, Wt::WPointF source, Wt::WPointF sink);

	Wt::Signal<int>& selectNodeSignal() { return _selectNodeSignal; };

	Wt::Signal<>& updateCanvas() { return _updateCanvas; }


protected:
	double mZoomFactor;
	Wt::WPointF mLastMousePos;
	Wt::WPointF mLastDelta;

	bool isDragging = false;
	bool canMoveNode = false;
	bool reorderFlag = true;
	bool mEditingEnabled = true;
	bool drawLineFlag = false;

	Wt::WPointF mTranslate = Wt::WPointF(0, 0);
	Wt::WPointF mMousePoint = Wt::WPointF(0, 0);

	std::shared_ptr<dyno::SceneGraph> mScene;

	Wt::Signal<int> _selectNodeSignal;
	Wt::Signal<> _updateCanvas;
};