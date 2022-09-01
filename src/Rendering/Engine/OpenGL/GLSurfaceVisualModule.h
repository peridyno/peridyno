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
#include "GLCudaBuffer.h"
#include "gl/VertexArray.h"
#include "gl/Program.h"

namespace dyno
{
	class GLSurfaceVisualModule : public GLVisualModule
	{
		DECLARE_CLASS(GLSurfaceVisualModule)
	public:
		GLSurfaceVisualModule();

	public:
		std::string caption() override;

		//bool faceNormal = true;

		DEF_VAR(bool, UsePhongShadingModel, false, "")

		DEF_INSTANCE_IN(TriangleSet<DataType3f>, TriangleSet, "");

		DEF_ARRAY_IN(Vec3f, Color, DeviceType::GPU, "");

		DEF_VAR(uint, ColorMode, 0, "");

	protected:
		virtual void paintGL(RenderPass mode) override;
		virtual void updateGL() override;
		virtual bool initializeGL() override;

	private:

		gl::Program mShaderProgram;
		gl::VertexArray	mVAO;

		GLCudaBuffer	mVertexBuffer;
		GLCudaBuffer	mNormalBuffer;
		GLCudaBuffer 	mIndexBuffer;
		GLCudaBuffer	mColor;

		DArray<Vec3f> mColorBuffer;

		unsigned int	mDrawCount = 0;
		unsigned int	mColorMode = 0;
	};
};