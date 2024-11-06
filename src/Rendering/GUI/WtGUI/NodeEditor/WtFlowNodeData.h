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

private:
	Wt::WPointF	_origin;

	mutable Wt::WRectF _boundingRect;
};


