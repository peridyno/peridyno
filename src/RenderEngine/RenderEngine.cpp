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
	}	   

	void RenderEngine::render(const RenderParams& rparams)
	{
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