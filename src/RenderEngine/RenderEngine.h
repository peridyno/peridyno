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
#include "gl/Buffer.h"
#include "RenderParams.h"

//use stb_image load image
#include "ui/picture.h"
#include <vector>

namespace dyno
{
	class SSAO;
	class ShadowMap;
	class RenderHelper;

	class Camera;

	class SceneGraph;
	class RenderTarget;
	//HE Xiaowei
	class RenderParams;

	struct Picture;
	
	enum CameraType
	{
		Orbit = 0,
		TrackBall
	};

	class RenderEngine
	{
	public:
		RenderEngine();
		~RenderEngine();

		void initialize(int width, int height, float scale);

		void setupCamera();

		void begin();
		void end();

		void draw(dyno::SceneGraph* scene);

		void drawGUI();

		void resizeRenderTarget(int w, int h);

		RenderParams* renderParams() { return mRenderParams; }
		RenderTarget* renderTarget() { return mRenderTarget; }

		std::shared_ptr<Camera> camera() { return mCamera; }

		bool cameraLocked();
	private:
		void initUniformBuffers();

	private:
		bool mDisenableCamera = false;
		// bool mOpenCameraRotate = true;

		glm::vec4 mClearColor = glm::vec4(0.45f, 0.55f, 0.60f, 1.00f);

		// Save pictrue's texture ID
		//std::vector<std::shared_ptr<Picture>> pics;

		// uniform buffer for matrices
		gl::Buffer		mTransformUBO;
		gl::Buffer		mLightUBO;
		
		SSAO*			mSSAO;
		ShadowMap*		mShadowMap;
		RenderHelper*	mRenderHelper;

		//HE Xiaowei
		RenderTarget* mRenderTarget;
		RenderParams* mRenderParams;

		std::shared_ptr<Camera> mCamera;
		CameraType mCameraType = CameraType::Orbit;

		std::vector<std::shared_ptr<Picture>> mPics;
	};
};
