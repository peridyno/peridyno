/**
 * Copyright 2022 Xukun Luo
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
#include "Node.h"
#include "Node/ParametricModel.h"

#include "Primitive/Primitive3D.h"
#include "Topology/DiscreteElements.h"

#include "Field/FilePath.h"

#include "ofbx.h"
#include "Topology/TextureMesh.h"
#include "Topology/TriangleSet.h"
#include "Topology/PolygonSet.h"
#include "Topology/HierarchicalModel.h"
#define REFTIME 46186158000L


namespace dyno
{
	/*!
	*	\class	SkeletonLoader
	*	\brief	Load a Skeleton 
	*/
	
	
	template<typename TDataType>
	class SkeletonLoader : public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(SkeletonLoader, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;


		SkeletonLoader();
		virtual ~SkeletonLoader();

	public:

		/**
		* @brief FBX file
		*/
		DEF_VAR(FilePath, FileName, "", "");
		DEF_VAR(bool, ImportAnimation, true, "ImportAnimation");

			/**
		* @brief Capsule Topology
		*/	
		DEF_INSTANCE_STATE(TopologyModule, Topology, "Topology");
		DEF_INSTANCE_STATE(TextureMesh, TextureMesh, "TextureMesh");


		DEF_INSTANCE_STATE(PolygonSet<TDataType>,PolygonSet,"PolygonSet");
		DEF_INSTANCE_STATE(HierarchicalScene, HierarchicalScene, "ModelObjects");

		DEF_INSTANCE_STATE(EdgeSet<TDataType>, JointSet, "TextureMesh");
		//DEF_INSTANCE_STATE(PointSet<TDataType>, ShapeCenter, "");

		DEF_VAR(bool,UseInstanceTransform,true,"reset pivot by bounding");

		DEF_VAR(Real, Radius, 0.0075, " Radius of Capsule");
		


	protected:

		bool initFBX();

		void varAnimationChange();

		void resetStates() override;

		void updateStates() override;

		void updateTransform();

		void updateAnimation(float time);

	private:


		void updateHierarchicalScene(const std::vector<std::shared_ptr<MeshInfo>>& meshsInfo, const std::vector< std::shared_ptr<Bone>>& bonesInfo);

		void setBonesToScene(const std::vector< std::shared_ptr<Bone>>& bonesInfo, std::shared_ptr<HierarchicalScene> scene);

		void setMeshToScene(const std::vector<std::shared_ptr<MeshInfo>>& meshsInfo, std::shared_ptr<HierarchicalScene> scene);

		bool loadTexture(const char* path, dyno::CArray2D<dyno::Vec4f>& img);

		void updateTextureMesh(const std::vector<std::shared_ptr<MeshInfo>>& meshsInfo);

		std::shared_ptr<Bone> pushBone(const ofbx::Object* bone, std::map<std::string, std::string>& parentTag, std::map<std::string, std::shared_ptr<ModelObject>>& name_ParentObj, std::vector<std::shared_ptr<Bone>>& bonesInfo, float scale);
		
		void buildHierarchy(const std::map<std::string, std::string>& obj_parent, const std::map<std::string, std::shared_ptr<ModelObject>>& name_ParentObj);

		void getCurveValue(const ofbx::AnimationCurveNode* node, std::shared_ptr<HierarchicalScene> scene,float scale);

		void coutName_Type(ofbx::Object::Type ty, ofbx::Object* obj);

		Mat4f getNodeTransformMatrix() 
		{
			Mat4f nodeRotation = this->computeQuaternion().toMatrix4x4();;
			Vec3f nodeLocation = this->varLocation()->getValue();
			Vec3f nodeSale = this->varScale()->getValue();
			Mat4f scaleMatrix = Mat4f(nodeSale[0], 0, 0, 0, 0, nodeSale[1], 0, 0, 0, 0, nodeSale[2], 0, 0, 0, 0, 1);

			Mat4f nodeTransform = Mat4f(
				nodeRotation(0, 0) * nodeSale[0], nodeRotation(0, 1), nodeRotation(0, 2), nodeLocation[0],
				nodeRotation(1, 0), nodeRotation(1, 1) * nodeSale[1], nodeRotation(1, 2), nodeLocation[1],
				nodeRotation(2, 0), nodeRotation(2, 1), nodeRotation(2, 2) * nodeSale[2], nodeLocation[2],
				nodeRotation(3, 0), nodeRotation(3, 1), nodeRotation(3, 2), nodeRotation(3, 3)) * scaleMatrix;
			return nodeTransform;
		}


	
	private:

		DArray<Vec3f> initialPosition;
		DArray<Vec3f> initialNormal;
		
		std::vector<Vec3f> initialShapeCenter;
	};
}