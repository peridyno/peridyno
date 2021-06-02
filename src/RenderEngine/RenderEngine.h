#pragma once

#include "RenderParams.h"

#include <vector>

#include <vtkOpenGLRenderer.h>
#include <vtkExternalOpenGLRenderWindow.h>
#include <vtkExternalOpenGLCamera.h>

namespace dyno
{
	class SceneGraph;
	class VtkVisualModule;
	class RenderEngine
	{
	public:
		RenderEngine();
		~RenderEngine();

		void initialize();
		void render(const RenderParams& rparams);	
		void setSceneGraph(dyno::SceneGraph* scene);

	private:
		void clearActors();

	private:

		vtkOpenGLRenderer*				m_vtkRenderer;
		vtkExternalOpenGLRenderWindow*	m_vtkWindow;
		vtkExternalOpenGLCamera*		m_vtkCamera;

	private:
		dyno::SceneGraph*				m_sceneGraph;
		
	};
};
