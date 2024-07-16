#pragma once

#include "WtNodeGraphicsObject.h"
#include "WtNodeDataModel.h"
#include <memory>

class NodeGeometry
{
public:
	NodeGeometry(std::unique_ptr<WtNodeDataModel> const& dataModel);
};