#pragma once

#include <RenderEngine.h>


#include <vtkExternalOpenGLRenderer.h>
#include <vtkExternalOpenGLRenderWindow.h>
#include <ExternalVTKWidget.h>

namespace dyno
{
	class VtkRenderEngine : public RenderEngine
	{
	public:
		VtkRenderEngine();
		virtual void draw(dyno::SceneGraph* scene) override; 
		virtual void resize(int w, int h) override;

	private:
		void setScene(dyno::SceneGraph* scene);

	private:

		vtkExternalOpenGLRenderer*		renderer = NULL;
		vtkExternalOpenGLRenderWindow*	window = NULL;

		vtkNew<ExternalVTKWidget>	widget;

		dyno::SceneGraph* m_scene = NULL;
	};
};