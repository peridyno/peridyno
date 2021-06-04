#include "RenderEngine.h"
// dyno
#include "Framework/SceneGraph.h"
#include "Action/Action.h"
#include "module/VtkVisualModule.h"

#include <random>

#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkCylinderSource.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkAnnotatedCubeActor.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkPlaneSource.h>
#include <vtkCubeSource.h>

#include <array>

#include <glm/gtc/type_ptr.hpp>

namespace dyno
{
	// create vtkActor for the scene to render
	class CreateActor : public Action
	{
	public:
		CreateActor(vtkOpenGLRenderer* renderer)
		{
			m_renderer = renderer;
		}

	private:
		void process(Node* node) override
		{
			for (auto iter : node->getVisualModuleList())
			{
				auto m = std::dynamic_pointer_cast<VtkVisualModule>(iter);
				if (m)
				{
					//std::cout << "wtf" << std::endl;
					m_renderer->AddActor(m->createActor());
				}
			}
		}

		vtkOpenGLRenderer* m_renderer;
	};

	RenderEngine::RenderEngine()
	{
		m_sceneGraph = NULL;
	}

	RenderEngine::~RenderEngine()
	{

	}

	void RenderEngine::clearActors()
	{
		// clear actors first
		vtkActorCollection* actors = m_vtkRenderer->GetActors();
		actors->InitTraversal();
		//actors->Print(std::cout);
		vtkActor* actor = actors->GetNextActor();
		while (actor != NULL)
		{
			//actor->Print(std::cout);
			m_vtkRenderer->RemoveActor(actor);
			actor = actors->GetNextActor();
		}

		// add back some actor
		m_vtkRenderer->AddActor(m_planeActor);
		m_vtkRenderer->AddActor(m_bboxActor);
	}

	void RenderEngine::setSceneGraph(dyno::SceneGraph* scene)
	{
		clearActors();		

		m_sceneGraph = scene;

		if (m_sceneGraph != NULL)
		{
			auto root = m_sceneGraph->getRootNode();
			root->traverseTopDown<CreateActor>(m_vtkRenderer);
		}
	}

	void RenderEngine::initialize()
	{
		m_vtkWindow = vtkExternalOpenGLRenderWindow::New();
		m_vtkWindow->AutomaticWindowPositionAndResizeOff();

		m_vtkRenderer = vtkOpenGLRenderer::New();
		m_vtkRenderer->GradientBackgroundOn();
		m_vtkWindow->AddRenderer(m_vtkRenderer);

		m_vtkCamera = vtkExternalOpenGLCamera::New();
		m_vtkRenderer->SetActiveCamera(m_vtkCamera);

		// ground plane 
		vtkNew<vtkPlaneSource> planeSource;
		planeSource->SetXResolution(20);
		planeSource->SetYResolution(20);
		vtkNew<vtkPolyDataMapper> planeMapper;
		planeMapper->SetInputConnection(planeSource->GetOutputPort());

		m_planeActor->SetMapper(planeMapper);
		m_planeActor->GetProperty()->SetRepresentationToWireframe();
		m_planeActor->RotateX(90);
		m_planeActor->SetPosition(0.5, 0, 0.5);
		m_planeActor->SetScale(2);

		// bounding box
		vtkNew<vtkCubeSource> cubeSource;
		cubeSource->SetBounds(0, 1, 0, 1, 0, 1);
		vtkNew<vtkPolyDataMapper> cubeMapper;
		cubeMapper->SetInputConnection(cubeSource->GetOutputPort());

		m_bboxActor->SetMapper(cubeMapper);
		m_bboxActor->GetProperty()->SetRepresentationToWireframe();
		m_bboxActor->GetProperty()->SetLineWidth(2);

		m_vtkRenderer->AddActor(m_bboxActor);
		m_vtkRenderer->AddActor(m_planeActor);
				
	}	   

	void RenderEngine::render(const RenderParams& rparams)
	{
		m_planeActor->SetVisibility(rparams.showGround);
		m_bboxActor->SetVisibility(rparams.showSceneBounds);

		// set background
		m_vtkRenderer->SetBackground(rparams.bgColor0[0], rparams.bgColor0[1], rparams.bgColor0[2]);
		m_vtkRenderer->SetBackground2(rparams.bgColor1[0], rparams.bgColor1[1], rparams.bgColor1[2]);

		// update viewport
		m_vtkWindow->SetSize(rparams.viewport.w, rparams.viewport.h);

		const float* temp;
		// update camera
		double view[16] = { 0.0 };
		temp = (const float*)glm::value_ptr(rparams.view);
		for (int i = 0; i < 16; ++i) view[i] = temp[i];
		m_vtkCamera->SetViewTransformMatrix(view);

		double proj[16] = { 0.0 };
		temp = (const float*)glm::value_ptr(rparams.proj);
		for (int i = 0; i < 16; ++i) proj[i] = temp[i];
		m_vtkCamera->SetProjectionTransformMatrix(proj);

		m_sceneGraph->draw();
		m_vtkWindow->Render();
	}

}