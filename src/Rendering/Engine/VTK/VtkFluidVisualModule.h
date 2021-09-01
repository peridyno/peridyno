#pragma once

#include "VtkVisualModule.h"
#include "Topology/PointSet.h"

namespace dyno
{
	class VtkFluidVisualModule : public VtkVisualModule
	{
		DECLARE_CLASS(VtkFluidVisualModule)
	public:
		VtkFluidVisualModule();
		
		DEF_INSTANCE_IN(PointSet<DataType3f>, PointSet, "");
	};
};