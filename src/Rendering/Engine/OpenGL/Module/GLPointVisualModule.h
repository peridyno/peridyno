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
#include "Topology/PointSet.h"

#include "GLVisualModule.h"
#include "gl/GPUBuffer.h"
#include "gl/VertexArray.h"
#include "gl/Shader.h"


namespace dyno
{
	class GLPointVisualModule : public GLVisualModule
	{
		DECLARE_CLASS(GLPointVisualModule)
	public:


		DECLARE_ENUM(ColorMapMode,
			PER_OBJECT_SHADER = 0,	// use constant color
			PER_VERTEX_SHADER = 1
		);

		GLPointVisualModule();
		~GLPointVisualModule();

		void setColorMapMode(ColorMapMode mode);

	public:
#ifdef CUDA_BACKEND
		DEF_INSTANCE_IN(PointSet<DataType3f>, PointSet, "");
#endif

#ifdef VK_BACKEND
		DEF_INSTANCE_IN(PointSet, PointSet, "");
#endif // VK_BACKEND


		DEF_ARRAY_IN(Vec3f, Color, DeviceType::GPU, "");

	public:
		DEF_VAR(float, PointSize, 0.001f, "Size of rendered particles");

		DEF_ENUM(ColorMapMode, ColorMode, ColorMapMode::PER_OBJECT_SHADER, "Color Mode");

	protected:
		virtual void updateGraphicsContext() override;

		virtual void paintGL(GLRenderPass pass) override;
		virtual void updateGL() override;
		virtual bool initializeGL() override;
		virtual void destroyGL() override;

	private:

		gl::VertexArray	mVertexArray;

		gl::XBuffer		mPosition;
		gl::XBuffer		mColor;

		unsigned int	mNumPoints;
		gl::Program*	mShaderProgram = 0;

	};
};
