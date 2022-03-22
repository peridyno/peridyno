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

		this->engine->mVisualModules.clear();
		// enqueue render content
		if (scene != nullptr && !scene->isEmpty())
		{
			scene->traverseForward(this);
		}
	}

	void process(Node* node) override
	{
		for (auto iter : node->graphicsPipeline()->activeModules())
		{
			auto m = dynamic_cast<VtkVisualModule*>(iter);
			if (m)
			{
				this->engine->mVisualModules.push_back(m);
			}
		}
	}

	VtkRenderEngine* engine;
};

dyno::VtkRenderEngine::VtkRenderEngine()
{
	// initialize vtk window and renderer
	mVtkRenderer->SetActiveCamera(mVtkCamera);
	mVtkRenderer->SetPreserveDepthBuffer(false);
	mVtkRenderer->SetPreserveColorBuffer(false);
	mVtkRenderer->SetPreserveGLLights(false);
	mVtkRenderer->SetPreserveGLCameraMatrices(false);
	mVtkRenderer->SetGradientBackground(true);
			
	mVtkWindow->AutomaticWindowPositionAndResizeOff();
	mVtkWindow->SetUseExternalContent(false);
	mVtkWindow->SetOffScreenRendering(m_useOffScreen);
	mVtkWindow->SwapBuffersOff();
	mVtkWindow->DoubleBufferOff();
	mVtkWindow->SetMultiSamples(0);

	// Toggle the line smoothing on, otherwise error occurs when calling the GetProperty()->SetLineWidth(1.5) function 
	mVtkWindow->LineSmoothingOn();

	mVtkWindow->AddRenderer(mVtkRenderer);
	
	// light
	mVtkLight->SetLightTypeToSceneLight();
	mVtkLight->PositionalOff();
	mVtkRenderer->AddLight(mVtkLight);
	
	// create a ground plane
	{
		float scale = 2.0;

		mPlane->SetOrigin(-scale, 0.0,  scale);
		mPlane->SetPoint1( scale, 0.0,  scale);
		mPlane->SetPoint2(-scale, 0.0, -scale);
		mPlane->SetResolution(100, 100);
		mPlane->Update();

		vtkNew<vtkPolyDataMapper> planeMapper;
		planeMapper->SetInputData(mPlane->GetOutput());

		mPlaneActor->SetMapper(planeMapper);
		mPlaneActor->GetProperty()->SetEdgeVisibility(true);
		mPlaneActor->GetProperty()->SetEdgeColor(0.4, 0.4, 0.4);
		mPlaneActor->GetProperty()->SetColor(0.8, 0.8, 0.8);
		mPlaneActor->GetProperty()->SetBackfaceCulling(true);
		//m_planeActor->GetProperty()->SetOpacity(0.5);

		mPlaneWireFrame->SetOrigin(-scale, 0.0, scale);
		mPlaneWireFrame->SetPoint1(scale, 0.0, scale);
		mPlaneWireFrame->SetPoint2(-scale, 0.0, -scale);
		mPlaneWireFrame->SetResolution(5, 5);
		mPlaneWireFrame->Update();

		vtkNew<vtkPolyDataMapper> wireframeMapper;
		wireframeMapper->SetInputData(mPlaneWireFrame->GetOutput());

		mPlaneWireFrameActor->SetMapper(wireframeMapper);
		mPlaneWireFrameActor->GetProperty()->SetEdgeVisibility(true);
		mPlaneWireFrameActor->GetProperty()->SetEdgeColor(0.35, 0.35, 0.35);
		mPlaneWireFrameActor->GetProperty()->SetColor(0.8, 0.8, 0.8);
		mPlaneWireFrameActor->GetProperty()->SetBackfaceCulling(true);
		mPlaneWireFrameActor->GetProperty()->SetFrontfaceCulling(true);
		mPlaneWireFrameActor->GetProperty()->SetLineWidth(1.5);

		mVtkRenderer->AddActor(mPlaneActor);
		mVtkRenderer->AddActor(mPlaneWireFrameActor);
	}

	// create a scene bounding box
	{
		vtkNew<vtkPolyDataMapper> mapper;
		mapper->SetInputData(mSceneCube->GetOutput());

		// wireframe
		mBoxActor->SetMapper(mapper);
		mBoxActor->GetProperty()->SetRepresentationToWireframe();
		mBoxActor->GetProperty()->SetColor(0.8, 0.8, 0.8);
		mBoxActor->GetProperty()->SetOpacity(0.8);
		// m_bboxActor->GetProperty()->SetLineWidth(2.0);
		mBoxActor->GetProperty()->SetLighting(false);
		
		mVtkRenderer->AddActor(mBoxActor);

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
		mRenderPasses.baker->SetResolution(4096);
		mRenderPasses.shadow->SetShadowMapBakerPass(mRenderPasses.baker);

		mRenderPasses.passes->AddItem(mRenderPasses.baker);
		mRenderPasses.passes->AddItem(mRenderPasses.shadow);
		//m_renderPasses.passes->AddItem(m_renderPasses.light);
		mRenderPasses.passes->AddItem(mRenderPasses.translucent);
		mRenderPasses.passes->AddItem(mRenderPasses.volume);
		mRenderPasses.passes->AddItem(mRenderPasses.overlay);
		
		mRenderPasses.seq->SetPasses(mRenderPasses.passes);
		mRenderPasses.cameraPass->SetDelegatePass(mRenderPasses.seq);

		// tell the renderer to use our render pass pipeline
		//m_vtkRenderer->SetPass(m_renderPasses.cameraPass);

	}
}

void VtkRenderEngine::initialize(int width, int height)
{
	mCamera = std::make_shared<OrbitCamera>();

	mCamera->setWidth(width);
	mCamera->setHeight(height);
	mCamera->registerPoint(float(width) / 2, float(height) / 2);
	mCamera->translateToPoint(0, 0);

	mCamera->zoom(3.0f);
	mCamera->setClipNear(0.01f);
	mCamera->setClipFar(10.0f);
}

void dyno::VtkRenderEngine::draw(dyno::SceneGraph * scene)
{
	mVtkWindow->GetState()->ResetFramebufferBindings();
	mVtkWindow->GetState()->ResetGLViewportState();

	setScene(scene);
	setCamera();

	// set light
	{
		glm::vec3 lightClr = m_rparams.light.mainLightColor;
		glm::vec3 lightDir = m_rparams.light.mainLightDirection * 100.f;		
		glm::vec3 ambient = m_rparams.light.ambientColor * m_rparams.light.ambientScale;		

		mVtkLight->SetColor(lightClr.r, lightClr.g, lightClr.b);
		mVtkLight->SetAmbientColor(ambient.r, ambient.g, ambient.b);
		mVtkLight->SetPosition(lightDir.x, lightDir.y, lightDir.z);
		//m_vtkLight->SetIntensity(m_rparams.light.mainLightScale);
	}

	// set bounding box
	{
		auto b0 = scene->getLowerBound();
		auto b1 = scene->getUpperBound();
		mSceneCube->SetBounds(b0[0], b1[0], b0[1], b1[1], b0[2], b1[2]);
		mSceneCube->Update();
	}
	
	// background
	{
		glm::vec3 color0 = m_rparams.bgColor0;
		glm::vec3 color1 = m_rparams.bgColor1;
		mVtkRenderer->SetBackground(color0.x, color0.y, color0.b);
		mVtkRenderer->SetBackground2(color1.x, color1.y, color1.b);
	}

	// update nodes
	for (auto* item : mVisualModules)
	{
		if (item->isVisible())
			item->updateRenderingContext();
		
		if (item->getActor())
			item->getActor()->SetVisibility(item->isVisible());
		if (item->getVolume())
			item->getVolume()->SetVisibility(item->isVisible());
	}

	mPlaneActor->SetVisibility(m_rparams.showGround);
	mPlaneWireFrameActor->SetVisibility(m_rparams.showGround);

	mBoxActor->SetVisibility(m_rparams.showSceneBounds);

	// with vtk_glew.h, we directly use OpenGL functions here
	GLint currFBO;
	glGetIntegerv(GL_FRAMEBUFFER_BINDING, &currFBO);

	mVtkWindow->Render();

	// blit offscreen vtk framebuffer to screen
	if(m_useOffScreen) {
		vtkOpenGLFramebufferObject* offscreen = mVtkWindow->GetOffScreenFramebuffer();
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

std::string VtkRenderEngine::name()
{
	return std::string("VTK");
}

void VtkRenderEngine::setScene(dyno::SceneGraph * scene)
{
	if (scene == m_scene)
		return;
		
	m_scene = scene;

	// first clear existing actors
	for (dyno::VtkVisualModule* item : mVisualModules) {
		vtkActor* actor = item->getActor();
		if (actor != NULL)
		{			
			mVtkRenderer->RemoveActor(actor);
		}

		vtkVolume* volume = item->getVolume();
		if (volume != NULL)
		{
			mVtkRenderer->RemoveVolume(volume);
		}
	}
	
	// gather vtk actors and volumes
	GatherVisualModuleAction action(this, scene);

	// add to renderer...
	for (dyno::VtkVisualModule* item : mVisualModules)
	{
		vtkActor* actor = item->getActor();
		if (actor != NULL)
		{		
			mVtkRenderer->AddActor(actor);
		}

		vtkVolume* volume = item->getVolume();
		if (volume != NULL)
		{
			mVtkRenderer->AddVolume(volume);
		}
	}	
}


void dyno::VtkRenderEngine::setCamera()
{
	// setup camera
	auto camera = RenderEngine::camera();
	
	glm::dmat4 view = camera->getViewMat();
	glm::dmat4 proj = camera->getProjMat();

	mVtkCamera->SetViewTransformMatrix(glm::value_ptr(view));
	mVtkCamera->SetProjectionTransformMatrix(glm::value_ptr(proj));

	// set window size..
	mVtkWindow->SetSize(camera->viewportWidth(), camera->viewportHeight());	
}
