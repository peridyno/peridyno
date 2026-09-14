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
	class GLDiscreteElementJointVisualModule : public GLVisualModule
	{
		DECLARE_TCLASS(GLDiscreteElementJointVisualModule, TDataType);
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename ::dyno::BallAndSocketJoint<Real> BallAndSocketJoint;
		typedef typename ::dyno::SliderJoint<Real> SliderJoint;
		typedef typename ::dyno::HingeJoint<Real> HingeJoint;
		typedef typename ::dyno::FixedJoint<Real> FixedJoint;
		typedef typename ::dyno::PointJoint<Real> PointJoint;
		typedef typename ::dyno::DistanceJoint<Real> DistanceJoint;

		GLDiscreteElementJointVisualModule();

		virtual std::string caption() override;

	public:
		DEF_INSTANCE_IN(DiscreteElements<TDataType>, DiscreteElements, "");

		// Joint visualization parameters
		DEF_VAR(Real, Length, 1.0f, "Base size of the joint visualization");
		DEF_VAR(Real, ThicknessScale, 1.0f, "Global thickness scale multiplier");

		// Joint visibility
		DEF_VAR(bool, ShowHingeJoint, true, "Show hinge joints");
		DEF_VAR(bool, ShowSliderJoint, true, "Show slider joints");
		DEF_VAR(bool, ShowBallAndSocketJoint, true, "Show ball and socket joints");
		DEF_VAR(bool, ShowFixedJoint, true, "Show fixed joints");
		DEF_VAR(bool, ShowPointJoint, true, "Show point joints");
		DEF_VAR(bool, ShowDistanceJoint, true, "Show distance joints");

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

		// Instance transforms and colors (GPU)
		DArray<Transform3f> mJointTransforms;
		DArray<Vec3f> mJointColors;
		// Sphere-shaped joint instances (e.g. ball of ball-and-socket joint)
		// rendered with Sphere template mesh instead of Box template
		DArray<Transform3f> mJointSphereTransforms;
		DArray<Vec3f> mJointSphereColors;

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

		InstancePass mJointPass;        // Box template: connectors, hinge axis, slider rails, fixed/point/distance joints
		InstancePass mJointSpherePass;  // Sphere template: ball of ball-and-socket joint

		// Surface rendering resources (shader + uniform blocks for instance passes)
		Program*	mShaderProgram = nullptr;
		Buffer		mRenderParamsUBlock;
		Buffer		mPBRMaterialUBlock;

		bool mTemplateLoaded = false;
	};

	IMPLEMENT_TCLASS(GLDiscreteElementJointVisualModule, TDataType);
};
