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
#include "GLCudaBuffer.h"
#include "gl/VertexArray.h"
#include "gl/Program.h"

namespace dyno
{
	class GLPointVisualModule : public GLVisualModule
	{
		DECLARE_CLASS(GLPointVisualModule)
	public:

		enum ColorMapMode
		{
			PER_OBJECT_SHADER = 0,	// use constant color
			PER_VERTEX_SHADER = 1
		};

		GLPointVisualModule();
		~GLPointVisualModule() override;

		void setPointSize(float size);
		float getPointSize() const;

		void setColorMapMode(ColorMapMode mode);
		void setColorMapRange(float vmin, float vmax);

	public:
		DEF_INSTANCE_IN(PointSet<DataType3f>, PointSet, "");

		DEF_ARRAY_IN(Vec3f, Color, DeviceType::GPU, "");

	protected:
		virtual void paintGL(RenderPass mode) override;
		virtual void updateGL() override;
		virtual bool initializeGL() override;

	private:
		unsigned int	mNumPoints;

		GLCudaBuffer	mPosition;
		GLCudaBuffer	mColor;

		float			mPointSize;
		gl::VertexArray	mVertexArray;

		gl::Program mShaderProgram;

		ColorMapMode	mColorMode = ColorMapMode::PER_OBJECT_SHADER;
		float			mColorMin = 0.f;
		float			mColorMax = 1.f;

		DArray<Vec3f> mColorBuffer;
	};
};
