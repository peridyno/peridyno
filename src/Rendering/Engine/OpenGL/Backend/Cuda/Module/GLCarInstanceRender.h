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
#include "GraphicsObject/VertexArray.h"
#include "GraphicsObject/Shader.h"
#include "GraphicsObject/GLTextureMesh.h"

#include "Topology/TextureMesh.h"

#ifdef CUDA_BACKEND
#include "ConstructTangentSpace.h"
#endif

namespace dyno
{
	class GLCarInstanceRender : public GLVisualModule
	{
		DECLARE_CLASS(GLCarInstanceRender)
	public:
		GLCarInstanceRender();
		~GLCarInstanceRender() override;

	public:
		virtual std::string caption() override;

		DEF_VAR(uint, MaterialShapeIndex, 0, "");
		DEF_INSTANCE_IN(TextureMesh, TextureMesh, "");
		DEF_ARRAYLIST_IN(Transform3f, Transform, DeviceType::GPU, "");
		DEF_ARRAYLIST_IN(float, HeadLight, DeviceType::GPU, "");
		DEF_ARRAYLIST_IN(float, BrakeLight, DeviceType::GPU, "");
		DEF_ARRAYLIST_IN(float, TurnSignal, DeviceType::GPU, "");
		DEF_VAR_IN(Vec3f, RightDirection, "");


	protected:
		void updateImpl() override;

		void paintGL(const RenderParams& rparams) override;
		void updateGL() override;
		bool initializeGL() override;
		void releaseGL() override;

	private:

		XBuffer<Vec3f> mTangent;
		XBuffer<Vec3f> mBitangent;

		XBuffer<Transform3f> mShapeTransform;

		Program*	mShaderProgram;
		Buffer		mRenderParamsUBlock;
		Buffer		mPBRMaterialUBlock;
		Buffer		mLightControlUBlock;

		VertexArray	mVAO;

		GLTextureMesh mTextureMesh;
		bool mNeedUpdateTextureMesh = false;

		CArray<uint> mOffset;
		CArray<List<Transform3f>> mLists;

		XBuffer<Transform3f> mXTransformBuffer;
		bool mNeedUpdateInstanceTransform = false;
		bool mNeedUpdateLight = false;

		XBuffer<float>		 mInstanceHeadlight;
		XBuffer<float>		 mInstanceBrakeLight;
		XBuffer<float>		 mInstanceTurnSignal;

#ifdef CUDA_BACKEND
		std::shared_ptr<ConstructTangentSpace> mTangentSpaceConstructor;
#endif
	};

};