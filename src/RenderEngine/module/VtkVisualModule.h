#pragma once

#include <Framework/ModuleVisual.h>

class vtkActor;
class vtkMapper;

namespace dyno
{
	class VtkVisualModule : public VisualModule
	{	
	public:
		VtkVisualModule();
		virtual ~VtkVisualModule();

		void setColor(float r, float g, float b, float a = 1.f);

		vtkActor* getActor();

	protected:
		// subclass should implement this method and call it in constructor
		virtual void createActor() = 0;

	protected:

		vtkActor*	m_actor = NULL;
		vtkMapper*	m_mapper = NULL;
	};
};