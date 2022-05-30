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

#include <Module/VisualModule.h>
#include <glm/vec3.hpp>

namespace dyno
{
	class GLVisualModule : public VisualModule
	{
	public:
		GLVisualModule();

		// basic Disney PBR material properties
		void setColor(const Vec3f& color);
		void setMetallic(float metallic);
		void setRoughness(float roughness);
		void setAlpha(float alpha);

		Vec3f getColor() const { return mBaseColor; }
		float getMetallic() const { return mMetallic; }
		float getRoughness() const { return mRoughness; }
		float getAlpha() const { return mAlpha; }

		virtual bool isTransparent() const;

	public:
		enum RenderPass
		{ 
			COLOR = 0,
			SHADOW = 1,
		};

		virtual bool initializeGL() = 0;
		virtual void updateGL() = 0;
		virtual void paintGL(RenderPass pass) = 0;

	protected:
		void updateGraphicsContext() final;

	private:
		bool isGLInitialized = false;

	protected:
		// material properties
		Vec3f			mBaseColor = Vec3f(0.8f);
		float			mMetallic = 0.0f;
		float			mRoughness = 0.5f;
		float			mAlpha = 1.f;		
	};
};