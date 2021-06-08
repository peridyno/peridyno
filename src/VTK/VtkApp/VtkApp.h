#pragma once

#include <vtkActor.h>
#include <vtkNew.h>

#include <vtkCubeSource.h>
#include <vtkPlaneSource.h>

#include <vtkOpenGLCamera.h>
#include <vtkOpenGLRenderer.h>
#include <vtkRenderWindow.h>

#include "../GUI/AppBase.h"

namespace dyno
{
	class VtkApp: public AppBase
	{
	public:
		VtkApp();
		~VtkApp();

		virtual void createWindow(int width, int height);
		virtual void mainLoop();

	private:

		vtkNew<vtkRenderWindow>			m_vtkWindow;
		vtkNew<vtkOpenGLRenderer>		m_vtkRenderer;
		vtkNew<vtkOpenGLCamera>			m_vtkCamera;

		// ground plane
		vtkNew<vtkPlaneSource>			m_scenePlane;
		vtkNew<vtkActor>				m_scenePlaneActor;
		// scene boundingbox
		vtkNew<vtkCubeSource>			m_sceneCube;
		vtkNew<vtkActor>				m_sceneCubeActor;		
		
	};
};
