#pragma once

#include "RenderParams.h"

#include <vector>

#include <vtkOpenGLRenderer.h>
#include <vtkExternalOpenGLRenderWindow.h>
#include <vtkExternalOpenGLCamera.h>
#include <vtkActor.h>
#include <vtkNew.h>


namespace dyno
{
	class SceneGraph;
	class VtkVisualModule;
	class RenderEngine
	{
	public:
		RenderEngine();
		~RenderEngine();

		// for external rendering
		void initializeExternal();
		void renderExternal(const RenderParams& rparams);	

		void setSceneGraph(dyno::SceneGraph* scene);

		void start();

	private:
		void clearActors();

	private:
		
		vtkNew<vtkOpenGLRenderer>		m_vtkRenderer;
		
		vtkNew<vtkActor>				m_planeActor;
		vtkNew<vtkActor>				m_bboxActor;


		// for external rendering
		vtkNew<vtkExternalOpenGLRenderWindow>	m_vtkWindow;
		vtkNew<vtkExternalOpenGLCamera>			m_vtkCamera;
		
	private:
		dyno::SceneGraph*				m_sceneGraph;
		
	};
};
