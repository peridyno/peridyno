#pragma once

#include <Framework/ModuleVisual.h>

#include <vtkTimeStamp.h>

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


		void display() final;
		void updateRenderingContext() final;
		
		bool isDirty(bool update = true);

	protected:

		vtkActor*	m_actor = NULL;
		vtkVolume*  m_volume = NULL;

		// timestamp for data sync
		vtkTimeStamp m_sceneTime;
		vtkTimeStamp m_updateTime;
	};
};