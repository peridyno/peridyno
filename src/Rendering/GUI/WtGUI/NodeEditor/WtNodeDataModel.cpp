#include "WtNodeDataModel.h"
#include "WtNodeStyle.h"

WtNodeDataModel::WtNodeDataModel()
	: _nodeStyle(WtStyleCollection::nodeStyle())
{
	// Derived classes can initialize specific style here
}

WtNodeStyle const& WtNodeDataModel::nodeStyle() const
{
	return _nodeStyle;
}

void WtNodeDataModel::setNodeStyle(WtNodeStyle const& style)
{
	_nodeStyle = style;
}