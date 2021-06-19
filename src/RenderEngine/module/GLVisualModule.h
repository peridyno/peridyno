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
	class GLVisualModule : public VisualModule
	{
	public:
		GLVisualModule();

		// override method from VisualModule
		void display() final;
		void updateRenderingContext() final;

		// basic disney PBR material properties
		void setColor(const glm::vec3& color);
		void setMetallic(float metallic);
		void setRoughness(float roughness);
		void setAlpha(float alpha);

		glm::vec3 getColor() const { return mBaseColor; }
		float getMetallic() const { return mMetallic; }
		float getRoughness() const { return mRoughness; }
		float getAlpha() const { return mAlpha; }

		virtual bool isTransparent() const;

	public:
		enum RenderMode
		{ 
			COLOR = 0,
			DEPTH = 1,
		};

		virtual bool initializeGL() = 0;
		virtual void updateGL() = 0;
		virtual void paintGL(RenderMode mode) = 0;

	private:
		bool isGLInitialized = false;

	protected:
		// material properties
		glm::vec3		mBaseColor = glm::vec3(0.8f);
		float			mMetallic = 0.0f;
		float			mRoughness = 0.5f;
		float			mAlpha = 1.f;		
	};
};