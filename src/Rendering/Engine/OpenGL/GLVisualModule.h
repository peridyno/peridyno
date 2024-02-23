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

#include <Color.h>
#include <RenderParams.h>
#include "Topology/TriangleSet.h"

namespace dyno
{
	// render pass
	struct GLRenderMode
	{
		const static int COLOR = 0;			// common color pass(opacity)
		const static int SHADOW = 1;		// shadow map pass
		const static int TRANSPARENCY = 2;	// transparency pass
	};

	class GLVisualModule : public VisualModule
	{
	public:
		GLVisualModule();
		~GLVisualModule() override;

		using GLTriangleSet = TriangleSet<DataType3f>;
		using GLPointSet = PointSet<DataType3f>;

		// basic Disney PBR material properties
		void setColor(const Color& color);
		void setMetallic(float metallic);
		void setRoughness(float roughness);
		void setAlpha(float alpha);

		virtual bool isTransparent() const;

		void draw(const RenderParams& rparams);

		// Attention: that this method should be called within OpenGL context
		void release();

	public:
		DEF_VAR(Color, BaseColor, Color(0.8f, 0.8f, 0.8f), "");
		DEF_VAR(Real, Metallic, 0.0f, "");
		DEF_VAR(Real, Roughness, 0.5f, "");
		DEF_VAR(Real, Alpha, 1.0f, "");

	protected:
		// override methods from Module
		virtual void updateImpl() override;

		// we use preprocess and postprocess method for update lock and timestamp
		virtual void preprocess() override final;
		virtual void postprocess() override final;

	protected:
		// methods for create/update/release OpenGL rendering content
		virtual bool initializeGL() = 0;
		virtual void releaseGL() = 0;
		virtual void updateGL() = 0;

		virtual void paintGL(const RenderParams& rparams) = 0;

	private:
		bool isGLInitialized = false;

		// mutex for sync data
		std::mutex	updateMutex;

		using clock = std::chrono::high_resolution_clock;
		// the timestamp when graphics context is changed by calling updateGraphicsContext
		clock::time_point changed;
		// the timestamp when GL resource is updated by updateGL
		clock::time_point updated;
	};
};