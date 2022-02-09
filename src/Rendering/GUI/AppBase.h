#pragma once

#include <vector>
#include <memory>

namespace dyno
{
	class ImWidget;
	class RenderEngine;
	class AppBase {
	public:
		AppBase() {};
		~AppBase() {};

		virtual void createWindow(int width, int height) {};
		virtual void mainLoop() = 0;

		virtual void setRenderEngine(std::shared_ptr<RenderEngine> engine) { mRenderEngine = engine; }
		virtual std::shared_ptr<RenderEngine> renderEngine() = 0;

	protected:
		std::shared_ptr<RenderEngine> mRenderEngine = nullptr;
	};

}
