#pragma once

#include "VtkVisualModule.h"
#include "Topology/PointSet.h"

namespace dyno
{
	class VtkPointVisualModule : public VtkVisualModule
	{
		DECLARE_CLASS(VtkPointVisualModule)
	public:
		VtkPointVisualModule();

		DEF_INSTANCE_IN(PointSet<DataType3f>, PointSet, "");
		DEF_ARRAY_IN(Vec3f, Color, DeviceType::GPU, "");
	};
};