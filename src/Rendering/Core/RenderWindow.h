/**
 * Copyright 2017-2023 Jian SHI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "RenderEngine.h"
#include "OrbitCamera.h"
#include "TrackballCamera.h"

namespace dyno
{
	// RenderWindow interface
	class RenderWindow
	{
	public:
		RenderWindow::RenderWindow()
		{
			// create a default camera
			mCamera = std::make_shared<OrbitCamera>();
			mCamera->setWidth(64);
			mCamera->setHeight(64);
			mCamera->registerPoint(0, 0);
			mCamera->rotateToPoint(-32, 12);
		}

		virtual void createWindow(int width, int height, bool usePlugin = false) {};
		virtual void mainLoop() = 0;

		virtual std::shared_ptr<RenderEngine> getRenderEngine() { return mRenderEngine; }
		virtual void setRenderEngine(std::shared_ptr<RenderEngine> engine) { mRenderEngine = engine; }

		virtual std::shared_ptr<Camera> getCamera() { return mCamera; }
		virtual void setCamera(std::shared_ptr<Camera> camera) { mCamera = camera; }

		RenderParams& getRenderParams() { return mRenderParams; }
		void		  setRenderParams(const RenderParams& rparams) { mRenderParams = rparams; }

		virtual void setWindowSize(int w, int h)
		{
			// TODO: handle viewport size by case
			mRenderParams.viewport.x = 0;
			mRenderParams.viewport.y = 0;
			mRenderParams.viewport.w = w;
			mRenderParams.viewport.h = h;

			mRenderParams.width = w;
			mRenderParams.height = h;

			mCamera->setWidth(w);
			mCamera->setHeight(h);
		}

	protected:
		std::shared_ptr<RenderEngine>	mRenderEngine;
		RenderParams					mRenderParams;

		std::shared_ptr<Camera>			mCamera;
	};
};
