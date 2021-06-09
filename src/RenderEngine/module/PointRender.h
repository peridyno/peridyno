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

#include "GLVisualModule.h"
#include "GLVertexArray.h"
#include "GLShader.h"

namespace dyno
{
	class PointRenderer : public GLVisualModule
	{
		DECLARE_CLASS(PointRenderer)
	public:
		PointRenderer();

		void setPointSize(float size);
		float getPointSize() const;

		bool isTransparent() const
		{
			return false;
		}

	protected:
		virtual void paintGL() override;
		virtual void updateGL() override;
		virtual bool initializeGL() override;

	private:
		unsigned int	mNumPoints;

		GLCudaBuffer	mPosition;
		GLCudaBuffer	mVelocity;
		GLCudaBuffer	mForce;

		float			mPointSize;
		GLVertexArray	mVertexArray;

		GLShaderProgram mShaderProgram;
	};
};
