#pragma once
#include <vector>
#include <algorithm>
#include <iostream>
#include <memory>
#include "SceneGraph.h"

namespace dyno
{
	class RenderEngine;
	class AppBase {
	public:
		AppBase(void) {};
		~AppBase() {};

		virtual void createWindow(int width, int height) {};
		virtual void mainLoop() = 0;

		virtual void setRenderEngine(RenderEngine* engine) { mRenderEngine = engine; }
	protected:
		RenderEngine* mRenderEngine;
	};

}
