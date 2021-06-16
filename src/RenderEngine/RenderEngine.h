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

#include "GLVertexArray.h"
#include "GLShader.h"
#include "GLTexture.h"
#include "RenderParams.h"

#include <vector>

class SSAO;
class ShadowMap;
class RenderHelper;
namespace dyno
{
	class SceneGraph;
	class RenderTarget;
	class RenderEngine
	{
	public:
		RenderEngine();
		~RenderEngine();

		void initialize();
		void draw(dyno::SceneGraph* scene, RenderTarget* target, const RenderParams& rparams);

	private:
		void initUniformBuffers();
		void renderSetup(dyno::SceneGraph* scene, RenderTarget* target, const RenderParams& rparams);

	private:
		// uniform buffer for matrices
		GLBuffer		mTransformUBO;
		GLBuffer		mShadowMapUBO;
		GLBuffer		mLightUBO;
		
		SSAO*			mSSAO;
		ShadowMap*		mShadowMap;
		RenderHelper*	mRenderHelper;

	};
};
