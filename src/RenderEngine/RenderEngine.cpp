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
#include <vtkCallbackCommand.h>

#include <array>

#include <glm/gtc/type_ptr.hpp>

namespace dyno
{
	// create vtkActor for the scene to render
	class ProcVisualModule : public Action
	{
	public:
		ProcVisualModule(vtkOpenGLRenderer* renderer)
		{
			m_renderer = renderer;
		}

	private:
		void process(Node* node) override
		{
			for (auto iter : node->getVisualModuleList())
			{
				auto module = std::dynamic_pointer_cast<VtkVisualModule>(iter);

				if (module)
				{
					vtkActor* actor = module->getActor();
					if(actor != NULL)
						m_renderer->AddActor(actor);

					vtkVolume* volume = module->getVolume();
					if (volume != NULL)
						m_renderer->AddVolume(volume);
				}
			}
		}

		vtkOpenGLRenderer* m_renderer;
	};

	RenderEngine::RenderEngine()
	{
		m_sceneGraph = NULL;


		// ground plane 
		m_scenePlane->SetXResolution(20);
		m_scenePlane->SetYResolution(20);
		vtkNew<vtkPolyDataMapper> planeMapper;
		planeMapper->SetInputConnection(m_scenePlane->GetOutputPort());

		m_scenePlaneActor->SetMapper(planeMapper);
		m_scenePlaneActor->GetProperty()->SetRepresentationToWireframe();
		//m_scenePlaneActor->RotateX(90);
		//m_scenePlaneActor->SetPosition(0.5, 0, 0.5);
		//m_scenePlaneActor->SetScale(2);

		// bounding box
		m_sceneCube->SetBounds(0, 1, 0, 1, 0, 1);
		vtkNew<vtkPolyDataMapper> cubeMapper;
		cubeMapper->SetInputConnection(m_sceneCube->GetOutputPort());

		m_sceneCubeActor->SetMapper(cubeMapper);
		m_sceneCubeActor->GetProperty()->SetRepresentationToWireframe();
		m_sceneCubeActor->GetProperty()->SetLineWidth(2);


		// renderer
		m_vtkRenderer->GradientBackgroundOn();
		m_vtkRenderer->SetBackground(0.4, 0.4, 0.4);
		m_vtkRenderer->SetBackground2(0.8, 0.8, 0.8);

		m_vtkRenderer->AddActor(m_sceneCubeActor);
		m_vtkRenderer->AddActor(m_scenePlaneActor);

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
			root->traverseTopDown<ProcVisualModule>(m_vtkRenderer);

			// update bounding box and ground plane based on scene bounds
			Vec3f bbox0 = m_sceneGraph->getLowerBound();
			Vec3f bbox1 = m_sceneGraph->getUpperBound();
			m_sceneCube->SetBounds(bbox0[0], bbox1[0], 
				bbox0[1], bbox1[1], 
				bbox0[2], bbox1[2]);

			float cx = (bbox0[0] + bbox1[0]) * 0.5f;
			float cy = (bbox0[1] + bbox1[1]) * 0.5f;
			float cz = (bbox0[2] + bbox1[2]) * 0.5f;
			
			float dx = bbox1[0] - bbox0[0];
			float dy = bbox1[1] - bbox0[1];
			float dz = bbox1[2] - bbox0[2];

			m_scenePlane->SetOrigin(cx - dx, bbox0[1], cz - dz);
			m_scenePlane->SetPoint1(cx + dx, bbox0[1], cz - dz);
			m_scenePlane->SetPoint2(cx - dx, bbox0[1], cz + dz);
		}

		// add back some actor
		m_vtkRenderer->AddActor(m_scenePlaneActor);
		m_vtkRenderer->AddActor(m_sceneCubeActor);
	}

	void RenderEngine::initializeExternal()
	{
		m_vtkWindow->AutomaticWindowPositionAndResizeOff();
		m_vtkWindow->AddRenderer(m_vtkRenderer);
		m_vtkRenderer->SetActiveCamera(m_vtkCamera);				
	}	   

	void RenderEngine::renderExternal(const RenderParams& rparams)
	{
		m_scenePlaneActor->SetVisibility(rparams.showGround);
		m_sceneCubeActor->SetVisibility(rparams.showSceneBounds);

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

	struct RuntimeData
	{
		SceneGraph* scene;
		bool run = false;
	};

	void timerCallbackFunc(vtkObject* caller, unsigned long eid, void* clientdata, void* calldata)
	{
		RuntimeData* data = static_cast<RuntimeData*>(clientdata);
		vtkRenderWindowInteractor *iren =
			static_cast<vtkRenderWindowInteractor*>(caller);

		if (data->run)
		{
			data->scene->takeOneFrame();
			iren->Render();
		}
	}

	void keyCallbackFunc(vtkObject* caller, unsigned long eid, void* clientdata, void* calldata)
	{
		RuntimeData* data = static_cast<RuntimeData*>(clientdata);		
		vtkRenderWindowInteractor *iren =
			static_cast<vtkRenderWindowInteractor*>(caller);

		if (iren->GetKeyCode() == 32)
		{
			// space key pressed
			data->run = !data->run;
			printf(data->run ? "Simulation start...\n": "Simulation stop...\n");
		}
	}

	void RenderEngine::start()
	{
		RuntimeData data;
		data.scene = m_sceneGraph;

		// TODO: move to App
		vtkNew<vtkRenderWindow> renderWindow;
		renderWindow->SetSize(1024, 768);
		renderWindow->AddRenderer(m_vtkRenderer);

		vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;
		renderWindowInteractor->SetRenderWindow(renderWindow);
		   
		renderWindowInteractor->Initialize();

		// use a timer for custom loop
		renderWindowInteractor->CreateRepeatingTimer(1);
		vtkNew<vtkCallbackCommand> timerCommand;
		timerCommand->SetClientData(&data);
		timerCommand->SetCallback(timerCallbackFunc);
		renderWindowInteractor->AddObserver(vtkCommand::TimerEvent, timerCommand);

		vtkNew<vtkCallbackCommand> keyCommand;
		keyCommand->SetClientData(&data);
		keyCommand->SetCallback(keyCallbackFunc);
		renderWindowInteractor->AddObserver(vtkCommand::KeyPressEvent, keyCommand);
		

		renderWindow->Render();
		renderWindowInteractor->Start();
	}

}