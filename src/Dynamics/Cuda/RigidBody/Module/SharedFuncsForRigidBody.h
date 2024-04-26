#pragma once
#include "Array/ArrayList.h"

#include "STL/Pair.h"

#include "Matrix/Transform3x3.h"

namespace dyno 
{
	void ApplyTransform(
		DArrayList<Transform3f>& instanceTransform,
		const DArray<Vec3f>& diff,
		const DArray<Vec3f>& translate,
		const DArray<Mat3f>& rotation,
		const DArray<Mat3f>& rotationInit,
		const DArray<Pair<uint, uint>>& binding,
		const DArray<int>& bindingtag);
}
