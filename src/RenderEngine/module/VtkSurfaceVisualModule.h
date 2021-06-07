#pragma once

#include "VtkVisualModule.h"

namespace dyno
{
	class SurfaceVisualModule : public VtkVisualModule
	{
		DECLARE_CLASS(SurfaceVisualModule)
	public:
		SurfaceVisualModule();

		virtual void updateRenderingContext() override;


	};
};