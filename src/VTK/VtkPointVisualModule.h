#pragma once

#include "VtkVisualModule.h"
#include "Topology/PointSet.h"

namespace dyno
{
	class PointVisualModule : public VtkVisualModule
	{
		DECLARE_CLASS(PointVisualModule)
	public:
		PointVisualModule();

		DEF_INSTANCE_IN(PointSet<DataType3f>, PointSet, "");
		DEF_ARRAY_IN(Vec3f, Color, DeviceType::GPU, "");
	};
};