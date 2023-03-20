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

#include <chrono>
#include <mutex>
#include <Module/VisualModule.h>

namespace dyno
{
	// render pass
	enum class GLRenderPass
	{
		COLOR  = 0,			// common color pass(opacity)
		SHADOW = 1,			// shadow map pass
		TRANSPARENCY = 2,	// transparency pass
	};

	class GLVisualModule : public VisualModule
	{
	public:
		GLVisualModule();
		~GLVisualModule();

		// basic Disney PBR material properties
		void setColor(const Vec3f& color);
		void setMetallic(float metallic);
		void setRoughness(float roughness);
		void setAlpha(float alpha);

		virtual bool isTransparent() const;

		void draw(GLRenderPass pass);

	public:
		DEF_VAR(Vec3f, BaseColor, Vec3f(0.8f), "");
		DEF_VAR(Real, Metallic, 0.0f, "");
		DEF_VAR(Real, Roughness, 0.5f, "");
		DEF_VAR(Real, Alpha, 1.0f, "");

	protected:
		virtual void updateGraphicsContext();

		virtual bool initializeGL() = 0;
		virtual void destroyGL() = 0;
		virtual void updateGL() = 0;
		virtual void paintGL(GLRenderPass pass) = 0;

		friend class GLRenderEngine;

	protected:
		bool isGLInitialized = false;
	
		using clock=std::chrono::high_resolution_clock;
		// the timestamp when graphics context is changed by calling updateGraphicsContext
		clock::time_point changed;
		// the timestamp when GL resource is updated by updateGL
		clock::time_point updated;

		// mutex for sync data
		std::mutex	updateMutex;
	};
};