#include "WtNodeGraphicsObject.h"

WtNodePainter::WtNodePainter() {}

WtNodePainter::~WtNodePainter() {}

void WtNodePainter::paint(Wt::WPainter* painter, WtNode& node)
{
	WtNodeGeometry const& geom = node.nodeGeometry();

	WtNodeState const& state = node.nodeState();

	WtNodeGraphicsObject const& graphicsObject = node.nodeGraphicsObject();

	WtNodeDataModel const* model = node.nodeDataModel();

	drawNodeRect(painter, geom, model, graphicsObject);
}

void WtNodePainter::drawNodeRect(
	Wt::WPainter* painter,
	WtNodeGeometry const& geom,
	WtNodeDataModel const* model,
	WtNodeGraphicsObject const& graphicsObject)
{
	WtNodeStyle const& nodeStyle = model->nodeStyle();
	//auto color = graphicsObject.isSelected() ? nodeStyle.SelectedBoundaryColor : nodeStyle.NormalBoundaryColor;
	auto color = nodeStyle.NormalBoundaryColor;
	if (geom.hovered())
	{
		Wt::WPen p(color);
		p.setWidth(nodeStyle.HoveredPenWidth);
		painter->setPen(p);
	}
	else
	{
		Wt::WPen p(color);
		p.setWidth(nodeStyle.PenWidth);
		painter->setPen(p);
	}

	float diam = nodeStyle.ConnectionPointDiameter;
	double const radius = 6.0;

	Wt::WRectF boundary = model->captionVisible() ? Wt::WRectF(-diam, -diam, 2.0 * diam + geom.width(), 2.0 * diam + geom.height())
		: Wt::WRectF(-diam, 0.0f, 2.0 * diam + geom.width(), diam + geom.height());

	//gradient

	if (model->captionVisible())
	{
		painter->drawRect(boundary);
	}
	else
	{
		painter->drawRect(boundary);
	}
}

void drawHotKeys(
	Wt::WPainter* painter,
	WtNodeGeometry const& geom,
	WtNodeDataModel const* model,
	WtNodeGraphicsObject const& graphicsObject)
{
	WtNodeStyle const& nodeStyle = model->nodeStyle();

	//auto color = graphicsObject.isSelected() ? nodeStyle.SelectedBoundaryColor : nodeStyle.NormalBoundaryColor;

	const Wt::WPen& pen = painter->pen();

	if (model->captionVisible() && model->hotkeyEnabled())
	{
		unsigned int captionHeight = geom.captionHeight();
		unsigned int keyWidth = geom.hotkeyWidth();
		unsigned int keyShift = geom.hotkeyIncline();
		unsigned int keyOffset = geom.hotkeyOffset();

		float diam = nodeStyle.ConnectionPointDiameter;

		//Wt different
		Wt::WRectF key0();

		double const radius = 6.0;

		//Wt different
		painter->setPen(Wt::WPen());
	}
}

//WtNodeGraphicsObject

WtNodeGraphicsObject::WtNodeGraphicsObject(WtFlowScene& scene, WtNode& node)
	: _scene(scene)
	, _node(node)
	, _locked(false)
{
}

WtNodeGraphicsObject::~WtNodeGraphicsObject() {}

//bool WtNodeGraphicsObject::isSelected()
//{
//	return _selected;
//}
//
//void WtNodeGraphicsObject::setSelected(bool selected)
//{
//	_selected = selected;
//}