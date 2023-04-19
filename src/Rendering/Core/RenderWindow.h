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

namespace dyno
{
	// RenderWindow interface
	class RenderWindow
	{
	public:
		RenderWindow();

		virtual void initialize(int width, int height) {}
		virtual void mainLoop() {}

		virtual std::shared_ptr<RenderEngine> getRenderEngine() { return mRenderEngine; }
		virtual void setRenderEngine(std::shared_ptr<RenderEngine> engine) { mRenderEngine = engine; }

		virtual std::shared_ptr<Camera> getCamera() { return mCamera; }
		virtual void setCamera(std::shared_ptr<Camera> camera) { mCamera = camera; }

		RenderParams& getRenderParams() { return mRenderParams; }
		void		  setRenderParams(const RenderParams& rparams) { mRenderParams = rparams; }

		virtual void setWindowSize(int w, int h);

	protected:
		std::shared_ptr<RenderEngine>	mRenderEngine;
		RenderParams					mRenderParams;

		std::shared_ptr<Camera>			mCamera;

	
	// interface for node picking
	public:
		// do region selection
		virtual const Selection& select(int x, int y, int w, int h);

		// set current selection (single)
		virtual void select(std::shared_ptr<Node> node, int instance = -1, int primitive = -1);

		virtual std::shared_ptr<Node> getCurrentSelectedNode();

	protected:
		// callback when node is picked by mouse event
		virtual void onSelected(const Selection& s);

		Selection selectedObject;

	};
};
