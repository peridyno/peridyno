#pragma once

#include "VtkVisualModule.h"
#include "Topology/PointSet.h"

namespace dyno
{
	class FluidVisualModule : public VtkVisualModule
	{
		DECLARE_CLASS(FluidVisualModule)
	public:
		FluidVisualModule();
		
		DEF_INSTANCE_IN(PointSet<DataType3f>, PointSet, "");
	};
};