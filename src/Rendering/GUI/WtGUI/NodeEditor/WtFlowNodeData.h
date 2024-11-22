#pragma once
#include <iostream>

#include <Wt/WPointF.h>
#include <Wt/WRectF.h>

#include "WtNodeData.hpp"

struct connectionPointData
{
	PortType portType;
	int id;
	PortShape portShape;
	Wt::WRectF pointRect;
	Wt::WPointF diamond_out[4];
	Wt::WPointF diamond[4];
};

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

	void setPointsData(std::vector<connectionPointData> pointsData) { _pointsData = pointsData; }
	std::vector<connectionPointData> getPointsData() { return _pointsData; }

private:

	Wt::WPointF	_origin;

	mutable Wt::WRectF _boundingRect;

	mutable Wt::WRectF _hotKey0BoundingRect;

	mutable Wt::WRectF _hotKey1BoundingRect;

	mutable std::vector<connectionPointData> _pointsData;
};
