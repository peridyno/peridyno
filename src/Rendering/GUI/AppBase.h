#pragma once

#include <vector>
#include <memory>

#include "SceneGraphFactory.h"
#include "RenderEngine.h"

namespace dyno
{
	class SceneGraph;
	class RenderWindow;

	class AppBase {
	public:
		AppBase() { std::cout << "\033[32m\033[1m" << ">>>: The program is loading necessary modules, it could take a while depending on your hardware... " << "\033[0m" << std::endl; }
		~AppBase() {};

		virtual void initialize(int width, int height, bool usePlugin = false) {};
		virtual void mainLoop() = 0;

		virtual RenderWindow* renderWindow() { return nullptr; }

		virtual std::shared_ptr<SceneGraph> getSceneGraph() { return SceneGraphFactory::instance()->active(); }
		virtual void setSceneGraph(std::shared_ptr<SceneGraph> scene) { SceneGraphFactory::instance()->pushScene(scene); }

		virtual void setSceneGraphCreator(std::function<std::shared_ptr<SceneGraph>()> creator) { SceneGraphFactory::instance()->setDefaultCreator(creator); }
	};
}
