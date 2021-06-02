#pragma once

#include "VtkVisualModule.h"

namespace dyno
{
	class SurfaceMapper;
	class SurfaceVisualModule : public VtkVisualModule
	{
		DECLARE_CLASS(SurfaceVisualModule)
	public:
		SurfaceVisualModule();

		virtual void updateRenderingContext() override;

	protected:
		vtkActor* createActor() override;

	private:
		SurfaceMapper*	m_mapper;
	};
};