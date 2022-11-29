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
#include "Topology/EdgeSet.h"

#include "GLVisualModule.h"
#include "gl/CudaBuffer.h"
#include "gl/VertexArray.h"
#include "gl/Shader.h"

namespace dyno
{
	class GLWireframeVisualModule : public GLVisualModule
	{
		DECLARE_CLASS(GLWireframeVisualModule)
	public:
		GLWireframeVisualModule();
	public:
		std::string caption() override;

		DEF_INSTANCE_IN(EdgeSet<DataType3f>, EdgeSet, "");

	protected:
		virtual void paintGL(GLRenderPass mode) override;
		virtual void updateGL() override;
		virtual bool initializeGL() override;

	private:
		// we use a cylinder to show line segment
		void createCylinder();
		
	private:

		gl::Program		mShaderProgram;

		gl::CudaBuffer	mPoints;
		gl::CudaBuffer 	mEdges;

		unsigned int	mNumEdges = 0;

		float			mRadius = 0.003f;

		// cylinder
		struct {
			gl::VertexArray	vao;
			gl::Buffer		vertices;
			gl::Buffer		normals;
			gl::Buffer		indices;
			unsigned int	drawCount;
			// number of sectors for create cylinder
			int				nSectors = 16;
		} mCylinder;
	};
};