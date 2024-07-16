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
#include "GraphicsObject/GPUBuffer.h"
#include "GraphicsObject/GPUTexture.h"
#include "GraphicsObject/VertexArray.h"
#include "GraphicsObject/Shader.h"

namespace dyno
{
	class GLSurfaceVisualModule : public GLVisualModule
	{
		DECLARE_CLASS(GLSurfaceVisualModule)
	public:
		GLSurfaceVisualModule();
		~GLSurfaceVisualModule() override;

	public:
		virtual std::string caption() override;

		DECLARE_ENUM(EColorMode,
			CM_Object = 0,
			CM_Vertex = 1,
			CM_Texture = 2);

		DEF_ENUM(EColorMode, ColorMode, EColorMode::CM_Object, "Color Mode");
		DEF_VAR(bool, UseVertexNormal, false, "");

#ifdef CUDA_BACKEND
		DEF_INSTANCE_IN(TriangleSet<DataType3f>, TriangleSet, "");
#endif

#ifdef  VK_BACKEND
		DEF_INSTANCE_IN(TriangleSet, TriangleSet, "");
#endif

		DEF_ARRAY_IN(Vec3f, Color,		DeviceType::GPU, "");
		DEF_ARRAY_IN(Vec3f, Normal,		DeviceType::GPU, "");		
		DEF_ARRAY_IN(Vec2f, TexCoord,	DeviceType::GPU, "");

		DEF_ARRAY_IN(TopologyModule::Triangle, NormalIndex, DeviceType::GPU, "");
		DEF_ARRAY_IN(TopologyModule::Triangle, TexCoordIndex, DeviceType::GPU, "");

#ifdef CUDA_BACKEND
		DEF_ARRAY2D_IN(Vec4f, ColorTexture, DeviceType::GPU, "");
		DEF_ARRAY2D_IN(Vec4f, BumpMap, DeviceType::GPU, "");
#endif

	protected:
		virtual void updateImpl() override;

		virtual void paintGL(const RenderParams& rparams) override;
		virtual void updateGL() override;
		virtual bool initializeGL() override;
		virtual void releaseGL() override;

	protected:

		Program*	mShaderProgram;

		// uniform blocks
		Buffer		mRenderParamsUBlock;
		Buffer		mPBRMaterialUBlock;

		VertexArray	mVAO;
		unsigned int	mNumTriangles = 0;

		XBuffer<Vec3f> mVertexPosition;
		XBuffer<Vec3f> mVertexColor;			// per-vertex color
		XBuffer<TopologyModule::Triangle> mVertexIndex;

		XBuffer<Vec3f> mNormal;
		XBuffer<TopologyModule::Triangle> mNormalIndex;

		XBuffer<Vec2f> mTexCoord;
		XBuffer<TopologyModule::Triangle> mTexCoordIndex;

#ifdef CUDA_BACKEND
		// color texture
		XTexture2D<Vec4f> mColorTexture;
		XTexture2D<Vec4f> mBumpMap;
#endif

		// for instanced rendering
		unsigned int			 mInstanceCount = 0;

	};
};