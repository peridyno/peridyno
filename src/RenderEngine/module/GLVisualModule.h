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

#include <Framework/ModuleVisual.h>
#include <glm/vec3.hpp>

namespace dyno
{
	class RenderEngine;
	class GLVisualModule : public VisualModule
	{
	public:
		enum ShadowMode
		{
			NONE = 0,	// do not cast/receive shadow
			CAST = 1,	// cast shadow
			RECV = 2,	// receive shadow
			ALL = 3,	// both...
		};

		enum ColorMapMode
		{
			CONSTANT = 0,	// use constant color
			VELOCITY_JET = 1,
			VELOCITY_HEAT = 2,
			FORCE_JET = 3,
			FORCE_HEAT = 4,
		};

	public:
		GLVisualModule();

		// override
		void display() final;
		void updateRenderingContext() final;

		// material properties
		void setColor(const glm::vec3& color);
		void setMetallic(float metallic);
		void setRoughness(float roughness);
		void setAlpha(float alpha);

		// colormap
		void setColorMapMode(ColorMapMode mode = CONSTANT);
		void setColorMapRange(float vmin, float vmax);

		// shadow mode
		void setShadowMode(ShadowMode mode);

		virtual bool isTransparent() const;

	protected:
		virtual bool initializeGL() = 0;
		virtual void updateGL() = 0;
		virtual void paintGL() = 0;

	private:
		bool isGLInitialized = false;

	protected:

		// material properties
		glm::vec3		mBaseColor = glm::vec3(0.8f);
		float			mMetallic = 0.5f;
		float			mRoughness = 0.5f;
		float			mAlpha = 1.f;

		// color map
		ColorMapMode	mColorMode = CONSTANT;
		float			mColorMin = 0.f;
		float			mColorMax = 1.f;

		ShadowMode		mShadowMode;

		friend class RenderEngine;
	};
};