#pragma once

#include "VtkVisualModule.h"
#include "Topology/TriangleSet.h"

namespace dyno
{
	class VtkSurfaceVisualModule : public VtkVisualModule
	{
		DECLARE_CLASS(VtkSurfaceVisualModule)
	public:
		VtkSurfaceVisualModule();

		DEF_INSTANCE_IN(TriangleSet<DataType3f>, TriangleSet, "");
	};
};