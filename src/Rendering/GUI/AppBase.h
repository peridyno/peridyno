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

		virtual void setRenderEngine(RenderEngine* engine) { mRenderEngine = engine; }
		virtual void addWidget(std::shared_ptr<ImWidget> widget) { mWidgets.push_back(widget); };

	protected:
		RenderEngine* mRenderEngine;

		std::vector<std::shared_ptr<ImWidget>> mWidgets;
	};

}
