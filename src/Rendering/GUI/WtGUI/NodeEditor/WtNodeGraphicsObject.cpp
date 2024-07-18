#include "WtNodeGraphicsObject.h"

WtNodePainter::WtNodePainter() {}

WtNodePainter::~WtNodePainter() {}

void WtNodePainter::paint(Wt::WPainter* painter)
{
	//drawNodeRect();

	WtNodeStyle nodeStyle;
	WtConnectionStyle connectStyle;
	WtFlowViewStyle flowStyle;
}

void WtNodePainter::drawNodeRect(Wt::WPainter* painter, NodeGeometry const& geom, WtNodeDataModel const* model, WtNodeGraphicsObject& graphicsObject)
{
	//Wt::WColor color = graphicsObject.isSelected() ? nodeStyle.SelectedBoundaryColor : nodeStyle.NormalBoundaryColor;
}

//WtNodeGraphicsObject

WtNodeGraphicsObject::WtNodeGraphicsObject() {}

WtNodeGraphicsObject::~WtNodeGraphicsObject() {}

bool WtNodeGraphicsObject::isSelected()
{
	return _selected;
}

void WtNodeGraphicsObject::setSelected(bool selected)
{
	_selected = selected;
}