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

#include "GraphicsObject/Buffer.h"
#include "GraphicsObject/Framebuffer.h"
#include "GraphicsObject/Texture.h"
#include "GraphicsObject/Shader.h"
#include "GraphicsObject/Mesh.h"

#include <vector>
#include <RenderEngine.h>

namespace dyno
{
	class Camera;
	class SceneGraph;

	class ShadowMap
	{
	public:
		ShadowMap(int size = 1024);
		~ShadowMap();

		void initialize();
		void release();

		void update(dyno::SceneGraph* scene, const dyno::RenderParams& rparams);

		// bind uniform block and texture
		void bind(int shadowUniformLoc = 3, int shadowTexSlot = 5);

		int  getSize() const;
		void setSize(int size);

		int  getNumBlurIterations() const;
		void setNumBlurIterations(int iter);
		void setLightRadius(float radius);
		float getLightRadius();


	private:
		// framebuffers
		Framebuffer		mFramebuffer;
		Texture2D		mShadowTex;
		Texture2D		mShadowDepth;
		Texture2D		mShadowBlur;

		Program*		mBlurProgram;
		Mesh*			mQuad;


		Buffer			mShadowUniform;	// uniform buffer for shadow lookup matrices

		bool			sizeUpdated = false;
		int				size = -1;

		// num of blur interations for VSM
		int				blurIters = 1;
		float			mLightRadius = 1;

	public:

		// patch to color bleeding, min p_max
		float			minValue = 0.1f;

	};
}