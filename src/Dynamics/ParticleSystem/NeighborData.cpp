#include "NeighborData.h"
#include "DataTypes.h"
#include "Topology/FieldNeighbor.h"
namespace dyno
{
	template class NeighborField<int>;
	template class NeighborField<TPair<DataType3f>>;
	template class NeighborField<TPair<DataType3d>>;
}