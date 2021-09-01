#include "VtkRenderEngine.h"

#include <iostream>

#include <SceneGraph.h>
#include <Action.h>

#include <OrbitCamera.h>
#include <TrackballCamera.h>

#include <glm/gtc/type_ptr.hpp>

#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>

#include <vtkLight.h>
#include <vtkLightActor.h>
#include <vtkLightCollection.h>
#include <vtkTextureObject.h>

#include <vtkOpenGLFramebufferObject.h>
#include <vtkOpenGLState.h>

#include <vtk_glew.h>

using namespace dyno;

struct dyno::GatherVisualModuleAction : public Action
{
	GatherVisualModuleAction(VtkRenderEngine* engine, dyno::SceneGraph* scene)
	{
		this->engine = engine;

		// enqueue render content
		if ((scene != 0) && (scene->getRootNode() != 0))
		{
			scene->getRootNode()->traverseTopDown(this);
		}
	}

	void process(Node* node) override
	{
		this->engine->m_modules.clear();

		for (auto iter : node->graphicsPipeline()->activeModules())
		{
			auto m = dynamic_cast<VtkVisualModule*>(iter);
			if (m)
			{
				this->engine->m_modules.push_back(m);
			}
		}
	}

	VtkRenderEngine* engine;
};

dyno::VtkRenderEngine::VtkRenderEngine()
{
	// initialize vtk window and renderer
	m_vtkRenderer->SetActiveCamera(m_vtkCamera);
	m_vtkRenderer->SetPreserveDepthBuffer(false);
	m_vtkRenderer->SetPreserveColorBuffer(false);
	m_vtkRenderer->SetPreserveGLLights(false);
	m_vtkRenderer->SetPreserveGLCameraMatrices(false);
	m_vtkRenderer->SetGradientBackground(true);
			
	m_vtkWindow->AutomaticWindowPositionAndResizeOff();
	m_vtkWindow->SetUseExternalContent(false);
	m_vtkWindow->SetOffScreenRendering(m_useOffScreen);
	m_vtkWindow->SwapBuffersOff();
	m_vtkWindow->DoubleBufferOff();
	m_vtkWindow->SetMultiSamples(0);

	m_vtkWindow->AddRenderer(m_vtkRenderer);
	
	// light
	m_vtkLight->SetLightTypeToSceneLight();
	m_vtkLight->PositionalOff();
	m_vtkRenderer->AddLight(m_vtkLight);
	
	// create a ground plane
	{
		float scale = 2.0;

		m_plane->SetOrigin(-scale, 0.0,  scale);
		m_plane->SetPoint1( scale, 0.0,  scale);
		m_plane->SetPoint2(-scale, 0.0, -scale);
		m_plane->SetResolution(100, 100);
		m_plane->Update();

		vtkNew<vtkPolyDataMapper> planeMapper;
		planeMapper->SetInputData(m_plane->GetOutput());

		m_planeActor->SetMapper(planeMapper);
		m_planeActor->GetProperty()->SetEdgeVisibility(true);
		m_planeActor->GetProperty()->SetEdgeColor(0.2, 0.2, 0.2);
		m_planeActor->GetProperty()->SetColor(0.8, 0.8, 0.8);
		m_planeActor->GetProperty()->SetBackfaceCulling(true);
		//m_planeActor->GetProperty()->SetOpacity(0.5);

		m_vtkRenderer->AddActor(m_planeActor);
	}

	// create a scene bounding box
	{
		vtkNew<vtkPolyDataMapper> mapper;
		mapper->SetInputData(m_sceneCube->GetOutput());

		// wireframe
		m_bboxActor->SetMapper(mapper);
		m_bboxActor->GetProperty()->SetRepresentationToWireframe();
		m_bboxActor->GetProperty()->SetColor(0.8, 0.8, 0.8);
		m_bboxActor->GetProperty()->SetOpacity(0.8);
		m_bboxActor->GetProperty()->SetLineWidth(2.0);
		m_bboxActor->GetProperty()->SetLighting(false);
		
		m_vtkRenderer->AddActor(m_bboxActor);

	}

	// set axes
	{
		// NOT WORK!
		//vtkNew<vtkRenderWindowInteractor> m_interactor;
		//m_interactor->SetRenderWindow(m_vtkWindow);

		//// axes
		//vtkNew<vtkAxesActor>				axes;
		//vtkNew<vtkOrientationMarkerWidget>	widget;

		//widget->SetOrientationMarker(axes);
		////widget->SetViewport(0.0, 0.0, 0.4, 0.4);

		//widget->SetInteractor(m_interactor);
		//widget->SetEnabled(1);
		//widget->InteractiveOn();

	}

	// set shadow pass
	{		
		m_renderPasses.baker->SetResolution(4096);
		m_renderPasses.shadow->SetShadowMapBakerPass(m_renderPasses.baker);

		m_renderPasses.passes->AddItem(m_renderPasses.baker);
		m_renderPasses.passes->AddItem(m_renderPasses.shadow);
		//m_renderPasses.passes->AddItem(m_renderPasses.light);
		m_renderPasses.passes->AddItem(m_renderPasses.translucent);
		m_renderPasses.passes->AddItem(m_renderPasses.volume);
		m_renderPasses.passes->AddItem(m_renderPasses.overlay);
		
		m_renderPasses.seq->SetPasses(m_renderPasses.passes);
		m_renderPasses.cameraPass->SetDelegatePass(m_renderPasses.seq);

		// tell the renderer to use our render pass pipeline
		//m_vtkRenderer->SetPass(m_renderPasses.cameraPass);

	}
}

void VtkRenderEngine::initialize(int width, int height, float scale)
{
	m_camera = std::make_shared<OrbitCamera>();

	m_camera->setWidth(width);
	m_camera->setHeight(height);
	m_camera->registerPoint(0.5f, 0.5f);
	m_camera->translateToPoint(0, 0);

	m_camera->zoom(3.0f);
	m_camera->setClipNear(0.01f);
	m_camera->setClipFar(10.0f);
}

void dyno::VtkRenderEngine::draw(dyno::SceneGraph * scene)
{
	//glEnable(GL_LIGHTING);
	//glEnable(GL_LIGHT0);

	setScene(scene);
	setCamera();
		

	// set light
	{
		glm::vec3 lightClr = m_rparams.light.mainLightColor;
		glm::vec3 lightDir = m_rparams.light.mainLightDirection * 100.f;		
		glm::vec3 ambient = m_rparams.light.ambientColor * m_rparams.light.ambientScale;		

		m_vtkLight->SetColor(lightClr.r, lightClr.g, lightClr.b);
		m_vtkLight->SetAmbientColor(ambient.r, ambient.g, ambient.b);
		m_vtkLight->SetPosition(lightDir.x, lightDir.y, lightDir.z);
		//m_vtkLight->SetIntensity(m_rparams.light.mainLightScale);
	}

	// set bounding box
	{
		auto b0 = scene->getLowerBound();
		auto b1 = scene->getUpperBound();
		m_sceneCube->SetBounds(b0[0], b1[0], b0[1], b1[1], b0[2], b1[2]);
		m_sceneCube->Update();
	}
	
	// background
	{
		glm::vec3 color0 = m_rparams.bgColor0;
		glm::vec3 color1 = m_rparams.bgColor1;
		m_vtkRenderer->SetBackground(color0.x, color0.y, color0.b);
		m_vtkRenderer->SetBackground2(color1.x, color1.y, color1.b);
	}

	// update nodes
	for (auto* item : m_modules)
	{
		if (item->isVisible())
			item->updateRenderingContext();
		
		if (item->getActor())
			item->getActor()->SetVisibility(item->isVisible());
		if (item->getVolume())
			item->getVolume()->SetVisibility(item->isVisible());
	}

	// with vtk_glew.h, we directly use OpenGL functions here
	GLint currFBO;
	glGetIntegerv(GL_FRAMEBUFFER_BINDING, &currFBO);

	m_vtkWindow->Render();

	// blit offscreen vtk framebuffer to screen
	if(m_useOffScreen) {
		vtkOpenGLFramebufferObject* offscreen = m_vtkWindow->GetOffScreenFramebuffer();
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, currFBO);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, offscreen->GetFBOIndex());
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		// TODO: we should not rely on camera width/height
		auto camera = RenderEngine::camera();
		int w = camera->viewportWidth();
		int h = camera->viewportHeight();
		glBlitFramebuffer(
			0, 0, w, h,
			0, 0, w, h,
			GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, GL_NEAREST);
	}

}


void dyno::VtkRenderEngine::resize(int w, int h)
{
	// TODO...
}

void dyno::VtkRenderEngine::setScene(dyno::SceneGraph * scene)
{
	if (scene == m_scene)
		return;
		
	m_scene = scene;

	// first clear existing actors
	for (dyno::VtkVisualModule* item : m_modules) {
		vtkActor* actor = item->getActor();
		if (actor != NULL)
		{			
			m_vtkRenderer->RemoveActor(actor);
		}

		vtkVolume* volume = item->getVolume();
		if (volume != NULL)
		{
			m_vtkRenderer->RemoveVolume(volume);
		}
	}
	
	// gather vtk actors and volumes
	GatherVisualModuleAction action(this, scene);

	// add to renderer...
	for (dyno::VtkVisualModule* item : m_modules)
	{
		vtkActor* actor = item->getActor();
		if (actor != NULL)
		{		
			m_vtkRenderer->AddActor(actor);
		}

		vtkVolume* volume = item->getVolume();
		if (volume != NULL)
		{
			m_vtkRenderer->AddVolume(volume);
		}
	}	
}


void dyno::VtkRenderEngine::setCamera()
{
	// setup camera
	auto camera = RenderEngine::camera();
	
	glm::dmat4 view = camera->getViewMat();
	glm::dmat4 proj = camera->getProjMat();

	m_vtkCamera->SetViewTransformMatrix(glm::value_ptr(view));
	m_vtkCamera->SetProjectionTransformMatrix(glm::value_ptr(proj));

	// set window size..
	m_vtkWindow->SetSize(camera->viewportWidth(), camera->viewportHeight());	
}
