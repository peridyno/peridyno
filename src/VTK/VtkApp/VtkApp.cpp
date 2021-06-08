#include "VtkApp.h"


#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkAnnotatedCubeActor.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkCallbackCommand.h>

#include "Framework/SceneGraph.h"
#include "Action/Action.h"
#include "../VtkVisualModule/VtkVisualModule.h"

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

	VtkApp::VtkApp()
	{
		// ground plane 
		m_scenePlane->SetXResolution(20);
		m_scenePlane->SetYResolution(20);
		vtkNew<vtkPolyDataMapper> planeMapper;
		planeMapper->SetInputConnection(m_scenePlane->GetOutputPort());

		m_scenePlaneActor->SetMapper(planeMapper);
		m_scenePlaneActor->GetProperty()->SetRepresentationToWireframe();

		// bounding box
		m_sceneCube->SetBounds(0, 1, 0, 1, 0, 1);
		vtkNew<vtkPolyDataMapper> cubeMapper;
		cubeMapper->SetInputConnection(m_sceneCube->GetOutputPort());

		m_sceneCubeActor->SetMapper(cubeMapper);
		m_sceneCubeActor->GetProperty()->SetRepresentationToWireframe();
		m_sceneCubeActor->GetProperty()->SetLineWidth(2);




	}

	VtkApp::~VtkApp()
	{

	}



	void VtkApp::createWindow(int width, int height)
	{
		auto root = SceneGraph::getInstance().getRootNode();
		root->traverseTopDown<ProcVisualModule>(m_vtkRenderer);

		// update bounding box and ground plane based on scene bounds
		Vec3f bbox0 = SceneGraph::getInstance().getLowerBound();
		Vec3f bbox1 = SceneGraph::getInstance().getUpperBound();
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
	
		m_vtkRenderer->GradientBackgroundOn();
		m_vtkRenderer->SetBackground(0.4, 0.4, 0.4);
		m_vtkRenderer->SetBackground2(0.8, 0.8, 0.8);

		m_vtkRenderer->AddActor(m_sceneCubeActor);
		m_vtkRenderer->AddActor(m_scenePlaneActor);

		SceneGraph::getInstance().draw();

		// initialize render window
		m_vtkWindow->SetSize(width, height);
		m_vtkWindow->AddRenderer(m_vtkRenderer);
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
			data->scene->draw();
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

	void VtkApp::mainLoop()
	{
		RuntimeData data;
		data.scene = &SceneGraph::getInstance();
		
		vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;
		renderWindowInteractor->SetRenderWindow(m_vtkWindow);
		   
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

		renderWindowInteractor->Start();
	}

}