#pragma once

#include <vector>
#include <memory>

namespace dyno
{
	class SceneGraph;

	class Camera;
	class RenderEngine;

	class AppBase {
	public:
		AppBase() {};
		~AppBase() {};

		virtual void createWindow(int width, int height, bool usePlugin = false) {};
		virtual void mainLoop() = 0;

		virtual std::shared_ptr<RenderEngine> getRenderEngine() { return mRenderEngine; }
		virtual void setRenderEngine(std::shared_ptr<RenderEngine> engine) { mRenderEngine = engine; }

		virtual std::shared_ptr<Camera> getCamera() { return mCamera; }
		virtual void setCamera(std::shared_ptr<Camera> camera) { mCamera = camera; }

		virtual std::shared_ptr<SceneGraph> getSceneGraph() { return mSceneGraph; }
		virtual void setSceneGraph(std::shared_ptr<SceneGraph> scene) { mSceneGraph = scene; }

	protected:
		std::shared_ptr<RenderEngine>	mRenderEngine;
		std::shared_ptr<Camera>			mCamera;
		std::shared_ptr<SceneGraph>		mSceneGraph;
	};

}
