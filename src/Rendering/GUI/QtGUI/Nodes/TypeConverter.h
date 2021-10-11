#pragma once

#include "BlockData.h"
#include "memory.h"

#include <functional>

namespace QtNodes
{

using SharedNodeData = std::shared_ptr<BlockData>;

// a function taking in NodeData and returning NodeData
using TypeConverter =
  std::function<SharedNodeData(SharedNodeData)>;

// data-type-in, data-type-out
using TypeConverterId =
  std::pair<BlockDataType, BlockDataType>;

}
