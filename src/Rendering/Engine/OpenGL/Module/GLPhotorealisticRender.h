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

#include <DeclarePort.h>
#include <Topology/TriangleSet.h>

#include "GLVisualModule.h"
#include "gl/GPUBuffer.h"
#include "gl/GPUTexture.h"
#include "gl/VertexArray.h"
#include "gl/Shader.h"
#include "gl/Shape.h"
#include "gl/Material.h"

namespace dyno
{
	class GLPhotorealisticRender : public GLVisualModule
	{
		DECLARE_CLASS(GLPhotorealisticRender)
	public:
		GLPhotorealisticRender();
		~GLPhotorealisticRender();

	public:
		virtual std::string caption() override;

		DECLARE_ENUM(EColorMode,
			CM_Object = 0,
			CM_Vertex = 1,
			CM_Texture = 2);

		DEF_ENUM(EColorMode, ColorMode, EColorMode::CM_Object, "Color Mode");

		DEF_VAR(bool, UseVertexNormal, false, "");

		DEF_ARRAY_IN(Vec3f, Vertex, DeviceType::GPU, "");

		DEF_ARRAY_IN(Vec3f, Color, DeviceType::GPU, "");

		DEF_ARRAY_IN(Vec3f, Normal,		DeviceType::GPU, "");
		
		DEF_ARRAY_IN(Vec2f, TexCoord,	DeviceType::GPU, "");


		DEF_INSTANCES_IN(gl::Shape, Shape, "");

		DEF_INSTANCES_IN(gl::Material, Material, "");

	protected:
		virtual void updateImpl() override;

		virtual void paintGL(const RenderParams& rparams) override;
		virtual void updateGL() override;
		virtual bool initializeGL() override;
		virtual void releaseGL() override;

	protected:

		gl::Program*	mShaderProgram;

		// uniform blocks
		gl::Buffer		mRenderParamsUBlock;
		gl::Buffer		mPBRMaterialUBlock;

		gl::VertexArray	mVAO;

		gl::XBuffer<Vec3f> mVertexPosition;
		gl::XBuffer<Vec3f> mVertexColor;			// per-vertex color

		gl::XBuffer<Vec3f> mNormal;

		gl::XBuffer<Vec2f> mTexCoord;

		// for instanced rendering
		unsigned int			 mInstanceCount = 0;

	};
};