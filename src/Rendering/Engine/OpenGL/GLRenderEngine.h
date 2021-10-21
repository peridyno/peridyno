/**
 * Copyright 2017-2021 Jian SHI
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

#include <memory>
#include <vector>

#include <Rendering.h>
#include "gl/Buffer.h"

namespace dyno
{
	class SSAO;
	class ShadowMap;
	class GLRenderHelper;

	class Camera;

	class SceneGraph;
	class GLRenderTarget;
	//HE Xiaowei

	struct Picture;
	
	enum CameraType
	{
		Orbit = 0,
		TrackBall
	};

	class GLRenderEngine : public RenderEngine
	{
	public:
		GLRenderEngine();
		~GLRenderEngine();
			   
		virtual void initialize(int width, int height) override;
		virtual void draw(dyno::SceneGraph* scene) override;
		virtual void resize(int w, int h) override;

		virtual std::string name() override;

	private:
		void setupCamera();
		void initUniformBuffers();

	private:

		// uniform buffer for matrices
		gl::Buffer		mTransformUBO;
		gl::Buffer		mLightUBO;
		
		SSAO*			mSSAO;
		ShadowMap*		mShadowMap;
		GLRenderHelper*	mRenderHelper;

		//HE Xiaowei
		GLRenderTarget* mRenderTarget;
		CameraType mCameraType = CameraType::Orbit;

	};
};
