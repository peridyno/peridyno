/**
 * Copyright 2017-2021 Xiaowei He (xiaowei@iscas.ac.cn)
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
#include "GLCudaBuffer.h"
#include "gl/VertexArray.h"
#include "gl/Program.h"

namespace dyno
{
	class GLInstanceVisualModule : public GLVisualModule
	{
		DECLARE_CLASS(GLInstanceVisualModule)
	public:
		GLInstanceVisualModule();

	public:
		std::string caption() override;

		DEF_INSTANCE_IN(TriangleSet<DataType3f>, TriangleSet, "");

		DEF_ARRAY_IN(Transform3f, Transform, DeviceType::GPU, "");

	protected:
		virtual void paintGL(RenderPass mode) override;
		virtual void updateGL() override;
		virtual bool initializeGL() override;

	private:

		gl::Program mShaderProgram;
		gl::VertexArray	mVAO;

		GLCudaBuffer	mVertexBuffer;
		GLCudaBuffer 	mIndexBuffer;
		GLCudaBuffer	mInstanceBuffer;

		unsigned int	mVertexCount = 0;
		unsigned int	mIndexCount = 0;

		unsigned int	mInstanceCount = 0;
	};
};