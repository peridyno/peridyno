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
		DEF_VAR(Real, Radius, 0.0075, " Radius of Capsule");
		DEF_VAR(bool, ImportAnimation, true, "ImportAnimation");
			/**
		* @brief Capsule Topology
		*/	
		DEF_INSTANCE_STATE(TopologyModule, Topology, "Topology");
		DEF_INSTANCE_STATE(TextureMesh, TextureMesh, "TextureMesh");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "TextureMesh");
		DEF_INSTANCE_STATE(PolygonSet<TDataType>,PolygonSet,"PolygonSet");
		DEF_INSTANCE_STATE(HierarchicalScene, HierarchicalScene, "ModelObjects");

		DEF_INSTANCE_STATE(EdgeSet<TDataType>, JointSet, "TextureMesh");


		DEF_VAR(bool,UseInstanceTransform,true,"reset pivot by bounding");

		/**
		* @brief FBX file
		*/
		DEF_VAR(FilePath, FileName, "", "");
		


	protected:

		bool initFBX();

		void varAnimationChange();

		void resetStates() override;

		void updateStates() override;

		void transform();

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
	};
}