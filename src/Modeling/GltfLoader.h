/**
 * Copyright 2022 Yuzhong Guo
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
#include "Node/ParametricModel.h"
#include "Topology/TriangleSet.h"
#include "Topology/TextureMesh.h"

#include "Module/ComputeModule.h"

#include "GLPointVisualModule.h"
#include "GLWireframeVisualModule.h"

#include "Field/FilePath.h"
#include "Topology/SkinInfo.h"
#include "Topology/JointInfo.h"


namespace dyno
{
	/**
	 * @brief A class to facilitate showing the shape information
	 */
	class BoundingBoxOfTextureMesh : public ComputeModule
	{
		DECLARE_CLASS(BoundingBoxOfTextureMesh);
	public:
		BoundingBoxOfTextureMesh();

		DEF_VAR(uint, ShapeId, 0, "");

		DEF_VAR(Vec3f, Center, Vec3f(0), "");

		DEF_VAR(Vec3f, LowerBound, Vec3f(0), "");

		DEF_VAR(Vec3f, UpperBound, Vec3f(0), "");

	public:
		DEF_INSTANCE_IN(TextureMesh, TextureMesh, "");

		DEF_INSTANCE_OUT(EdgeSet<DataType3f>, BoundingBox, "");

	private:
		void compute() override;

		void shapeIdChanged();
	};


	template<typename TDataType>
	class GltfLoader : virtual public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(GltfLoader, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename TopologyModule::Triangle Triangle;
		

		typedef unsigned char byte;
		typedef int joint;
		typedef int shape;
		typedef int mesh;
		typedef int primitive;
		typedef int scene;

		GltfLoader();
		~GltfLoader();

		std::string getNodeType() override { return "IO"; }

	public:

		DEF_VAR(FilePath, FileName, "", "");
		DEF_VAR(bool, ImportAnimation, false, "");
		DEF_VAR(Real,AnimationSpeed,1,"AnimationSpeed");
		DEF_VAR(Real, JointRadius, 0.01, "");

		DEF_VAR(bool, UseInstanceTransform, true, "");
		//DefaultChannel
		DEF_ARRAY_STATE(Vec2f, TexCoord_0, DeviceType::GPU, "UVSet 0");
		DEF_ARRAY_STATE(Vec2f, TexCoord_1, DeviceType::GPU, "UVSet 1");
		DEF_ARRAY_STATE(Mat4f, InitialMatrix, DeviceType::GPU, "");

		//CustomChannel

		DEF_VAR_STATE(Mat4f, Transform, Mat4f::identityMatrix(), "Transform");

		DEF_INSTANCE_STATE(SkinInfo, Skin, "SkinInfo");

		DEF_ARRAY_STATE(Mat4f, JointInverseBindMatrix, DeviceType::GPU, "JointInverseBindMatrix");
		DEF_ARRAY_STATE(Mat4f, JointLocalMatrix, DeviceType::GPU, "JointLocalMatrix");
		DEF_ARRAY_STATE(Mat4f, JointWorldMatrix, DeviceType::GPU, "JointWorldMatrix");
		
		DEF_INSTANCE_STATE(JointInfo,JointsData,"JointsInfo");
	
		DEF_INSTANCE_STATE(TextureMesh, TextureMesh, "");

		DEF_INSTANCE_STATE(PointSet<TDataType>, ShapeCenter, "");
		DEF_INSTANCE_STATE(EdgeSet<TDataType>, JointSet, "");

		DEF_INSTANCE_STATE(JointAnimationInfo, Animation,"");

	protected:
		void resetStates() override
		{
			updateTransform();
		}

		void updateStates() override;



	public:

		std::map<int, std::string> node_Name;

	private:

		DArray<Coord> initialPosition;
		DArray<Coord> unCenterPosition;
		DArray<Coord> initialNormal;
		DArray<int> d_joints;

		DArray<Coord> d_ShapeCenter;
		bool ToCenter = false;

		std::map<joint, Quat<float>> joint_rotation;
		std::map<joint, Vec3f> joint_scale;
		std::map<joint, Vec3f> joint_translation;
		std::map<joint, Mat4f> joint_matrix;
		std::map<joint, std::vector<int>> jointId_joint_Dir;





		std::map<joint, Mat4f> joint_inverseBindMatrix;
		std::map<joint, Mat4f> joint_AnimaMatrix;

		std::vector<std::string> Scene_Name;
		std::map<joint, std::string> joint_Name;

		std::map<joint, Vec3i> joint_output;		// Vec3i[0]  translation ,Vec3i[1]  scale ,Vec3i[2] rotation ,
		std::map<joint, Vec3f> joint_input;			// time Vec3f[0]  translation ,Vec3f[1]  scale ,Vec3f[2] rotation ,
		
		
		std::vector<joint> all_Joints;
		

		
		std::map<int, std::vector<int>> meshId_Dir;


		std::map<int, Mat4f> node_matrix;


		std::shared_ptr<GLWireframeVisualModule> jointLineRender;
		std::shared_ptr<GLPointVisualModule> jointPointRender;

		int maxMeshId = -1;
		int maxJointId = -1;
		
		

		DArray<Mat4f> d_mesh_Matrix;
		DArray<int> d_shape_meshId;


	private:


		void varChanged();

		void varRenderChanged();

		void varAnimation();

		void updateTransform();

		void updateAnimation(int frameNumber);

		void InitializationData();

		Vec3f getVertexLocationWithJointTransform(joint jointId, Vec3f inPoint, std::map<joint, Mat4f> jMatrix);

		void buildInverseBindMatrices(const std::vector<joint>& all_Joints);
		
		Vec3f getmeshPointDeformByJoint(joint jointId, Coord worldPosition, std::map<joint, Mat4f> jMatrix);

		void updateTransformState();
	};



	IMPLEMENT_TCLASS(GltfLoader, TDataType);
}