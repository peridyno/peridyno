#pragma once
#include <iostream>

#include <Wt/WPointF.h>
#include <Wt/WRectF.h>


class WtFlowNodeData
{
public:
	WtFlowNodeData();
	~WtFlowNodeData();

public:
	void setNodeOrigin(Wt::WPointF p) { _origin = p; }
	void setNodeOrigin(int x, int y) { _origin = Wt::WPointF(x, y); }
	Wt::WPointF getNodeOrigin() const { return _origin; }

	void setNodeBoundingRect(Wt::WRectF r) { _boundingRect = r; }
	Wt::WRectF getNodeBoundingRect() const { return _boundingRect; }

	void setHotKey0BoundingRect(Wt::WRectF r) { _hotKey0BoundingRect = r; }
	Wt::WRectF getHotKey0BoundingRect() { return _hotKey0BoundingRect; }

	void setHotKey1BoundingRect(Wt::WRectF r) { _hotKey1BoundingRect = r; }
	Wt::WRectF getHotKey1BoundingRect() { return _hotKey1BoundingRect; }

private:
	Wt::WPointF	_origin;

	mutable Wt::WRectF _boundingRect;

	mutable Wt::WRectF _hotKey0BoundingRect;

	mutable Wt::WRectF _hotKey1BoundingRect;
};


