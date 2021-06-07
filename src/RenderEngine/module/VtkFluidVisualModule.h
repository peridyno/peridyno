#pragma once

#include "VtkVisualModule.h"

namespace dyno
{
	class FluidVisualModule : public VtkVisualModule
	{
		DECLARE_CLASS(FluidVisualModule)
	public:
		FluidVisualModule();

		virtual void updateRenderingContext() override {}
	};
};