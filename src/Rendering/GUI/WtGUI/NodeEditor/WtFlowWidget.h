#pragma once

#include <Wt/WEvent.h>
#include <Wt/WPaintedWidget.h>
#include <Wt/WPointF.h>
#include <SceneGraph.h>

class WtFlowWidget : public Wt::WPaintedWidget
{
public:
	WtFlowWidget();
	virtual ~WtFlowWidget() = default;

	virtual void onMouseMove(const Wt::WMouseEvent& event) = 0;
	virtual void onMouseWentDown(const Wt::WMouseEvent& event) = 0;
	virtual void onMouseWentUp(const Wt::WMouseEvent& event) = 0;

	void onMouseWheel(const Wt::WMouseEvent& event);
	void zoomIn();
	void zoomOut();

	virtual void onKeyWentDown() = 0;

	void reorderNode();

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

	//std::shared_ptr<dyno::SceneGraph> mScene;

	Wt::Signal<int> _selectNodeSignal;
	Wt::Signal<> _updateCanvas;


};