#pragma once

#include <Framework/ModuleVisual.h>

class vtkActor;
class vtkVolume;

namespace dyno
{
	class VtkVisualModule : public VisualModule
	{	
	public:
		VtkVisualModule();
		virtual ~VtkVisualModule();

		void setColor(float r, float g, float b, float a = 1.f);

		vtkActor*	getActor();
		vtkVolume*	getVolume();

	protected:

		vtkActor*	m_actor = NULL;
		vtkVolume*  m_volume = NULL;
	};
};