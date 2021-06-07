#pragma once

#include "VtkVisualModule.h"

namespace dyno
{
	class PointVisualModule : public VtkVisualModule
	{
		DECLARE_CLASS(PointVisualModule)
	public:
		PointVisualModule();

		virtual void updateRenderingContext() override;
	};
};