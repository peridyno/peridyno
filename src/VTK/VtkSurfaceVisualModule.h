#pragma once

#include "VtkVisualModule.h"
#include "Topology/TriangleSet.h"

namespace dyno
{
	class SurfaceVisualModule : public VtkVisualModule
	{
		DECLARE_CLASS(SurfaceVisualModule)
	public:
		SurfaceVisualModule();

		DEF_INSTANCE_IN(TriangleSet<DataType3f>, TriangleSet, "");
	};
};