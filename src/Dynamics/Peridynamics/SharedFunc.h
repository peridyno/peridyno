#pragma once
#include "Array/Array.h"
#include "Array/ArrayList.h"

#include "NeighborData.h"
#include "DataTypes.h"

namespace dyno {
	template<typename Coord, typename NPair>
	void constructRestShape(
		DArrayList<NPair>& shape,
		DArrayList<int>& nbr,
		DArray<Coord>& pos);
}
