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
#include "Topology/TriangleSet.h"

#include "GLVisualModule.h"
#include "gl/GPUBuffer.h"
#include "gl/VertexArray.h"
#include "gl/Shader.h"

#include <DeclarePort.h>

namespace dyno
{
	class GLSurfaceVisualModule : public GLVisualModule
	{
		DECLARE_CLASS(GLSurfaceVisualModule)
	public:
		GLSurfaceVisualModule();
		~GLSurfaceVisualModule();

	public:
		virtual std::string caption() override;

		DECLARE_ENUM(EColorMode,
			CM_Object = 0,
			CM_Vertex = 1);

		DEF_ENUM(EColorMode, ColorMode, EColorMode::CM_Object, "Color Mode");

		DEF_VAR(bool, UseVertexNormal, false, "");

#ifdef CUDA_BACKEND
		DEF_INSTANCE_IN(TriangleSet<DataType3f>, TriangleSet, "");
#endif

#ifdef  VK_BACKEND
		DEF_INSTANCE_IN(TriangleSet, TriangleSet, "");
#endif

		DEF_ARRAY_IN(Vec3f, Color, DeviceType::GPU, "");

	protected:
		virtual void updateGraphicsContext() override;

		virtual void paintGL(GLRenderPass mode) override;
		virtual void updateGL() override;
		virtual bool initializeGL() override;
		virtual void destroyGL() override;

	protected:

		gl::Program*	mShaderProgram;
		gl::VertexArray	mVAO;

		gl::XBuffer		mIndexBuffer;
		gl::XBuffer		mVertexBuffer;
		gl::XBuffer		mNormalBuffer;
		gl::XBuffer		mColorBuffer;

		unsigned int	mDrawCount = 0;

		// for instanced rendering
		gl::XBuffer		mInstanceBuffer;
		unsigned int	mInstanceCount = 0;

	};
};