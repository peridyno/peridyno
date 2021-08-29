#include "VtkRenderEngine.h"
#include "VtkVisualModule.h"

#include <iostream>

#include <SceneGraph.h>
#include <Action.h>

using namespace dyno;

struct VtkVisualModules : public Action
{
private:
	void process(Node* node) override
	{
		for (auto iter : node->graphicsPipeline()->activeModules())
		{
			auto m = dynamic_cast<VtkVisualModule*>(iter);
			if (m && m->isVisible())
			{
				modules.push_back(m);
			}
		}
	}
public:
	std::vector<dyno::VtkVisualModule*> modules;
};

dyno::VtkRenderEngine::VtkRenderEngine()
{

}

void dyno::VtkRenderEngine::draw(dyno::SceneGraph * scene)
{
	// initialize vtk window and renderer
	if (window == NULL)
	{
		window = widget->GetRenderWindow(); 
		window->AutomaticWindowPositionAndResizeOff();

		renderer = widget->AddRenderer();
		renderer->SetBackground(1.0, 0.0, 0.0);

		window->AddRenderer(renderer);
	}
	// 
	setScene(scene);
	window->Render();
}

void dyno::VtkRenderEngine::resize(int w, int h)
{
	std::cout << w << "," << h << std::endl;
	window->SetSize(w, h);
}

void dyno::VtkRenderEngine::setScene(dyno::SceneGraph * scene)
{
	if (scene == m_scene)
	{
		return;
	}
		
	m_scene = scene;

	VtkVisualModules modules;
	// enqueue render content
	if ((scene != 0) && (scene->getRootNode() != 0))
	{
		scene->getRootNode()->traverseTopDown(&modules);
	}

	for (dyno::VtkVisualModule* item : modules.modules)
	{
		vtkActor* actor = item->getActor();
		if (actor != NULL)
		{
			std::cout << "vtkActor" << std::endl;
			renderer->AddActor(actor);
		}

		vtkVolume* volume = item->getVolume();
		if (volume != NULL)
		{
			std::cout << "vtkVolume" << std::endl;
			renderer->AddVolume(volume);
		}
	}

	renderer->ResetCamera();
}
