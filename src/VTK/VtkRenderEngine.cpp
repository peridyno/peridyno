#include "VtkRenderEngine.h"

#include <iostream>

#include <SceneGraph.h>
#include <Action.h>
#include <camera/Camera.h>

#include <glm/gtc/type_ptr.hpp>

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

	// TODO: light not work...
	m_vtkLight->SetDiffuseColor(1.0, 0.0, 0.0);
	m_vtkRenderer->AddExternalLight(m_vtkLight);

	m_vtkWindow->AutomaticWindowPositionAndResizeOff();
	m_vtkWindow->SetUseExternalContent(true);
	m_vtkWindow->AddRenderer(m_vtkRenderer);
}

void dyno::VtkRenderEngine::draw(dyno::SceneGraph * scene)
{
	// 
	setScene(scene);
	setCamera();

	// render params
	{
		glm::vec3 color0 = renderParams()->bgColor0;
		glm::vec3 color1 = renderParams()->bgColor1;

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

	m_vtkWindow->Render();
}


void dyno::VtkRenderEngine::setScene(dyno::SceneGraph * scene)
{
	if (scene == m_scene)
		return;
		
	m_scene = scene;
		
	GatherVisualModuleAction action(this, scene);

	for (dyno::VtkVisualModule* item : m_modules)
	{
		vtkActor* actor = item->getActor();
		if (actor != NULL)
		{
			std::cout << "vtkActor" << std::endl;			
			m_vtkRenderer->AddActor(actor);
		}

		vtkVolume* volume = item->getVolume();
		if (volume != NULL)
		{
			std::cout << "vtkVolume" << std::endl;
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
