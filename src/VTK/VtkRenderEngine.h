#pragma once

#include <RenderEngine.h>

#include <vtkExternalOpenGLRenderer.h>
#include <vtkExternalOpenGLRenderWindow.h>
#include <vtkExternalOpenGLCamera.h>
#include <vtkExternalLight.h>

#include <vtkActor.h>
#include <vtkCubeSource.h>
#include <vtkPlaneSource.h>

#include <vtkSequencePass.h>
#include <vtkDefaultPass.h>
#include <vtkLightsPass.h>
#include <vtkShadowMapPass.h>
#include <vtkShadowMapBakerPass.h>
#include <vtkRenderPassCollection.h>
#include <vtkCameraPass.h>
#include <vtkOverlayPass.h>
#include <vtkOpaquePass.h>
#include <vtkVolumetricPass.h>
#include <vtkTranslucentPass.h>

#include "VtkVisualModule.h"

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
		void setCamera();

	private:
		bool m_useOffScreen = true;

		vtkNew<vtkExternalOpenGLRenderer>		m_vtkRenderer;
		vtkNew<vtkExternalOpenGLRenderWindow>	m_vtkWindow;
		vtkNew<vtkExternalOpenGLCamera>			m_vtkCamera;

		vtkNew<vtkLight>						m_vtkLight;

		std::vector<dyno::VtkVisualModule*>		m_modules;

		// ground plane
		vtkNew<vtkPlaneSource>	m_plane;
		vtkNew<vtkActor>		m_planeActor;

		// bounding box
		vtkNew<vtkCubeSource>	m_sceneCube;
		vtkNew<vtkActor>		m_bboxActor;

		// render passes
		struct
		{
			vtkNew<vtkRenderPassCollection> passes;
			vtkNew<vtkSequencePass>			seq;

			vtkNew<vtkShadowMapPass>		shadow;
			vtkNew<vtkShadowMapBakerPass>	baker;
			vtkNew<vtkLightsPass>			light;
			vtkNew<vtkTranslucentPass>		translucent;
			vtkNew<vtkVolumetricPass>		volume;
			vtkNew<vtkOverlayPass>			overlay;
			vtkNew<vtkOpaquePass>			opaque;
			
			vtkNew<vtkCameraPass>			cameraPass;
		} m_renderPasses;


		dyno::SceneGraph* m_scene = NULL;

		friend struct GatherVisualModuleAction;
	};
};