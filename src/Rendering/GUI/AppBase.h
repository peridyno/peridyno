#pragma once

#include <vector>
#include <memory>

#include "SceneGraphFactory.h"
#include "Rendering.h"

namespace dyno
{
	class SceneGraph;
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

		virtual std::shared_ptr<SceneGraph> getSceneGraph() { return SceneGraphFactory::instance()->active(); }
		virtual void setSceneGraph(std::shared_ptr<SceneGraph> scene) { SceneGraphFactory::instance()->pushScene(scene); }

		RenderParams& getRenderParams() { return mRenderParams; }
		void		  setRenderParams(const RenderParams& rparams) { mRenderParams = rparams; }

		virtual void setWindowSize(int w, int h) 
		{
			// TODO: resize framebuffer out of render engine
			mRenderEngine->resize(w, h);

			mRenderParams.viewport.w = w;
			mRenderParams.viewport.h = h;

			mCamera->setWidth(w);
			mCamera->setHeight(h);			
		}

	protected:
		std::shared_ptr<RenderEngine>	mRenderEngine;
		RenderParams					mRenderParams;

		std::shared_ptr<Camera>			mCamera;
	};
}
