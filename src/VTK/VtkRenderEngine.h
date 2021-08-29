#pragma once

#include <RenderEngine.h>

#include <vtkExternalOpenGLRenderer.h>
#include <vtkExternalOpenGLRenderWindow.h>
#include <vtkExternalOpenGLCamera.h>
#include <vtkExternalLight.h>

#include "VtkVisualModule.h"

namespace dyno
{
	class VtkRenderEngine : public RenderEngine
	{
	public:
		VtkRenderEngine();
		virtual void draw(dyno::SceneGraph* scene) override; 

	private:
		void setScene(dyno::SceneGraph* scene);
		void setCamera();

	private:

		vtkNew<vtkExternalOpenGLRenderer>		m_vtkRenderer;
		vtkNew<vtkExternalOpenGLRenderWindow>	m_vtkWindow;
		vtkNew<vtkExternalOpenGLCamera>			m_vtkCamera;
		vtkNew<vtkExternalLight>				m_vtkLight;

		std::vector<dyno::VtkVisualModule*>		m_modules;

		dyno::SceneGraph* m_scene = NULL;

		friend struct GatherVisualModuleAction;
	};
};