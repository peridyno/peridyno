/**
 * Copyright 2026 Yuzhong Guo
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
#include "GraphicsObject/GPUBuffer.h"
#include "GraphicsObject/VertexArray.h"
#include "GraphicsObject/Shader.h"
#include "Topology/DiscreteElements.h"
#include "Topology/TriangleSet.h"
#include "Matrix/Transform3x3.h"

namespace dyno
{
	template<typename TDataType>
	class GLDiscreteElementVisualModule : public GLVisualModule
	{
		DECLARE_TCLASS(GLDiscreteElementVisualModule, TDataType);
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		GLDiscreteElementVisualModule();

		virtual std::string caption() override;

	public:
		DEF_INSTANCE_IN(DiscreteElements<TDataType>, DiscreteElements, "");

		// Element visibility
		DEF_VAR(bool, ShowBox, true, "Show Box3D elements");
		DEF_VAR(bool, ShowSphere, true, "Show Sphere3D elements");
		DEF_VAR(bool, ShowCapsule, true, "Show Capsule3D elements");
		DEF_VAR(bool, ShowTet, true, "Show Tet3D elements");
		DEF_VAR(bool, ShowTriangle, true, "Show Triangle3D elements");
		DEF_VAR(bool, ShowMedialCone, true, "Show MedialCone3D elements");
		DEF_VAR(bool, ShowMedialSlab, true, "Show MedialSlab3D elements");

	protected:
		virtual void updateImpl() override;

		virtual bool initializeGL() override;
		virtual void releaseGL() override;
		virtual void updateGL() override;
		virtual void paintGL(const RenderParams& rparams) override;

	private:
		// Template meshes for instance rendering
		TriangleSet<DataType3f> mStandardBox;
		TriangleSet<DataType3f> mStandardSphere;
		TriangleSet<DataType3f> mStandardCapsule;

		// Instance transforms and colors (GPU)
		DArray<Transform3f> mBoxTransforms;
		DArray<Vec3f> mBoxColors;
		DArray<Transform3f> mSphereTransforms;
		DArray<Vec3f> mSphereColors;
		DArray<Transform3f> mCapsuleTransforms;
		DArray<Vec3f> mCapsuleColors;

		// TriangleSet for MedialCone/MedialSlab/Tet/Triangle
		DArray<Vec3f> mTriVertices;
		DArray<Topology::Triangle> mTriIndices;

		// One render pass per instance group
		struct InstancePass
		{
			VertexArray vao;
			XBuffer<Topology::Triangle> vertexIndex;   // attribute 0 (triangle vertex indices)
			XBuffer<Vec3f> vertexPosition;             // SSBO binding 8 (template mesh)
			XBuffer<Transform3f> instanceTransform;   // attributes 3-7
			XBuffer<Vec3f> instanceColor;              // attribute 8
			unsigned int instanceCount = 0;
			unsigned int numTriangles = 0;
		};

		InstancePass mBoxPass;
		InstancePass mSpherePass;
		InstancePass mCapsulePass;

		// Surface rendering resources (for TriangleSet pass)
		Program*	mShaderProgram = nullptr;
		Buffer		mRenderParamsUBlock;
		Buffer		mPBRMaterialUBlock;
		VertexArray	mVAO;
		unsigned int	mNumTriangles = 0;
		XBuffer<Vec3f> mVertexPosition;
		XBuffer<Topology::Triangle> mVertexIndex;

		bool mTemplateLoaded = false;
	};

	IMPLEMENT_TCLASS(GLDiscreteElementVisualModule, TDataType);
};
