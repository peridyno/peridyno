#pragma once

#include <Module/VisualModule.h>

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


		virtual void display() final;
		virtual void updateRenderingContext() final;
		
		bool isDirty(bool update = true);

	protected:

		vtkActor*	m_actor = NULL;
		vtkVolume*  m_volume = NULL;

		// timestamp for data sync
		vtkTimeStamp m_sceneTime;
		vtkTimeStamp m_updateTime;
	};
};