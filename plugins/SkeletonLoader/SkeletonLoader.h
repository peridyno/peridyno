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
#include "Topology/JointTree.h"

#include "FilePath.h"

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
	class FbxMeshInfo : public ModelObject
	{
	public:
		FbxMeshInfo() {};

		std::vector<Vec3f> vertices;
		std::vector<int> verticeId_pointId;
		std::vector<Vec3f> normals;
		std::vector<Vec2f> texcoords;
		std::vector<Vec3f> verticesColor;
		std::vector<std::vector<TopologyModule::Triangle>> facegroup_triangles;
		std::vector<CArrayList<uint>> facegroup_polygons;

		std::vector<std::shared_ptr<Material>> materials;

		std::vector<TAlignedBox3D<Real>> boundingBox;
		std::vector<Transform3f> boundingTransform;
	};
	
	template<typename TDataType>
	class SkeletonLoader : public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(SkeletonLoader, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

        typedef std::vector<std::shared_ptr<JointTree<typename TDataType>>> JointList;

		SkeletonLoader();
		virtual ~SkeletonLoader();


		void loadFBX();
		bool initFBX(const char* filepath);


	public:
		DEF_VAR(Real, Radius, 0.0075 , " Radius of Capsule")
		/**
		* @brief Capsule Topology
		*/	
		DEF_INSTANCE_STATE(TopologyModule, Topology, "Topology");
		DEF_INSTANCE_STATE(TextureMesh, TextureMesh, "TextureMesh");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "TextureMesh");
		DEF_INSTANCE_STATE(PolygonSet<TDataType>,PolygonSet,"PolygonSet");
		DEF_INSTANCE_STATE(HierarchicalScene, HierarchicalScene, "ModelObjects");

		DEF_VAR(bool,UseInstanceTransform,true,"reset pivot by bounding");

		/**
		* @brief FBX file
		*/
		DEF_VAR(FilePath, FileName, "", "");
		
	public:

		int findMeshIDbyName(std::string name) 
		{
			int id = 0;
			for (auto it : mMeshs){
				if (it->name == name)
					return id;

				id++;
			}
			return -1;
		}



	protected:
		void resetStates() override;

		ofbx::IScene* mFbxScene = nullptr;

		std::vector<std::shared_ptr<FbxMeshInfo>> mMeshs;
		std::vector<std::shared_ptr<Bone>> mBones;

	private:

		void updateHierarchicalScene() 
		{
			auto hierarchicalScene = this->stateHierarchicalScene()->getDataPtr();

			for (auto it : mMeshs)
				hierarchicalScene->mModelObjects.push_back(it);

			for (auto it : mBones)
				hierarchicalScene->mModelObjects.push_back(it);

			const ofbx::GlobalSettings* settings = mFbxScene->getGlobalSettings();
			hierarchicalScene->mTimeStart = settings->TimeSpanStart;
			hierarchicalScene->mTimeEnd = settings->TimeSpanStop;

		}

		bool loadTexture(const char* path, dyno::CArray2D<dyno::Vec4f>& img);

		void updateTextureMesh() 
		{
			auto texMesh = this->stateTextureMesh()->getDataPtr();

			std::vector<int> mesh_VerticesNum;
			std::vector<int> mesh_NormalNum;
			std::vector<int> mesh_UvNum;

			std::vector<Vec3f> texVertices;
			std::vector<Vec3f> texNormals;
			std::vector<Vec2f> texCoords;

			for (auto it : mMeshs)
			{
				mesh_VerticesNum.push_back(it->vertices.size());
				mesh_NormalNum.push_back(it->normals.size());
				mesh_UvNum.push_back(it->texcoords.size());

				texVertices.insert(texVertices.end(), it->vertices.begin(), it->vertices.end());
				texNormals.insert(texNormals.end(), it->normals.begin(), it->normals.end());
				texCoords.insert(texCoords.end(), it->texcoords.begin(), it->texcoords.end());
			}

			texMesh->vertices().assign(texVertices);
			texMesh->normals().assign(texNormals);
			texMesh->texCoords().assign(texCoords);

			std::vector<uint> shapeID;
			shapeID.resize(texVertices.size());

			int tempID = 0;
			int offset = 0;
			for (size_t i = 0; i < mMeshs.size(); i++)
			{
				auto shape = std::make_shared<Shape>();

				int meshFaceGroupNum = mMeshs[i]->facegroup_triangles.size();
				for (size_t j = 0; j < meshFaceGroupNum; j++)
				{
					auto triangles = mMeshs[i]->facegroup_triangles[j];

					for (size_t k = 0; k < triangles.size(); k++)
					{
						triangles[k][0] = triangles[k][0] + offset;
						triangles[k][1] = triangles[k][1] + offset;
						triangles[k][2] = triangles[k][2] + offset;
						shapeID[triangles[k][0]] = tempID;
						shapeID[triangles[k][1]] = tempID;
						shapeID[triangles[k][2]] = tempID;
					}
					shape->vertexIndex.assign(triangles);
					shape->normalIndex.assign(triangles);
					shape->texCoordIndex.assign(triangles);

					tempID ++;

					texMesh->materials().push_back(mMeshs[i]->materials[j]);
					shape->material = mMeshs[i]->materials[j];
					shape->boundingBox = mMeshs[i]->boundingBox[j];
					shape->boundingTransform = mMeshs[i]->boundingTransform[j];


					texMesh->shapes().push_back(shape);
				}
				offset += mesh_VerticesNum[i];
			}
			
			texMesh->shapeIds().assign(shapeID);
		}



		void pushBone(const ofbx::Object* bone, std::map<std::string,std::string>& parentTag,std::map<std::string,std::shared_ptr<ModelObject>>& name_ParentObj)
		{
			auto it = parentTag.find(std::string(bone->name));

			if (it != parentTag.end()) {
				//std::cout << "Bone already exists : " << std::distance(parentTag.begin(), it) << std::endl;
			}
			else {

				auto temp = std::make_shared<Bone>();

				temp->name = bone->name;
				temp->preRotation = Vec3f(bone->getPreRotation().x, bone->getPreRotation().y, bone->getPreRotation().z);
				temp->localTranslation = Vec3f(bone->getLocalTranslation().x, bone->getLocalTranslation().y, bone->getLocalTranslation().z);
				temp->localRotation = Vec3f(bone->getLocalRotation().x, bone->getLocalRotation().y, bone->getLocalRotation().z);
				temp->localScale = Vec3f(bone->getLocalScaling().x, bone->getLocalScaling().y, bone->getLocalScaling().z);
				temp->pivot = Vec3f(bone->getRotationPivot().x, bone->getRotationPivot().y, bone->getRotationPivot().z);

				mBones.push_back(temp);


				if (bone->parent) 
				{
					parentTag[std::string(bone->name)] = std::string(bone->parent->name);
					name_ParentObj[std::string(bone->name)] = mBones[mBones.size() - 1];
				}
				else 
				{
					parentTag[std::string(bone->name)] = std::string("No parent object");
					name_ParentObj[std::string(bone->name)] = nullptr;
				}

				//std::cout << "pushBone : " << std::string(bone->name) << " parent : " <<parentTag[std::string(bone->name)] << "\n";


			}					
		}

		void buildHierarchy(const std::map<std::string, std::string>& obj_parent, const std::map<std::string,std::shared_ptr<ModelObject>>& name_ParentObj)
		{
			//std::cout << parentTag.size()<< " -- " << name_ParentObj.size() << "\n";
			for (auto it : obj_parent)
			{	
				//Parent
				std::pair<std::string, std::string> element = it;
				std::string objName = it.first;

				while (true)
				{
					if (element.second != std::string("No parent object")) 
					{
						std::string parentName = element.second;

						std::shared_ptr<ModelObject> obj = name_ParentObj.find(objName)->second;
						std::shared_ptr<ModelObject> parent = nullptr;

						auto parentIter = name_ParentObj.find(parentName);
						if(parentIter!= name_ParentObj.end())
							parent = parentIter->second;

						if (parent != nullptr)
						{
							obj->parent.push_back(parent);
							element = *obj_parent.find(parentName);
						}
						else
							break;
					}
					else
						break;
				}

				//Child
				auto parentIter = name_ParentObj.find(it.second);
				if (parentIter != name_ParentObj.end())
				{
					std::shared_ptr<ModelObject> parent = parentIter->second;
					parent->child.push_back(name_ParentObj.find(objName)->second);
				}
	
			}

		}

		std::shared_ptr<ModelObject> getModelObjectByName(std::string name)
		{
			for (size_t i = 0; i < mMeshs.size(); i++)
			{
				if (mMeshs[i]->name == name)
					return mMeshs[i];
			}
			for (size_t i = 0; i < mBones.size(); i++)
			{
				if (mBones[i]->name == name)
					return mBones[i];
			}
			return nullptr;
		}

		void getCurveValue(const ofbx::AnimationCurveNode* node) 
		{
			if (!node->getBone())
				return;

			auto bone = getModelObjectByName(std::string(node->getBone()->name));

			if (bone == nullptr)
				return;

			auto propertyData = node->getBoneLinkProperty();

			if (propertyData == "Lcl Translation")
			{
				auto x = node->getCurve(0);
				auto y = node->getCurve(1);
				auto z = node->getCurve(2);
				if (x) 
				{
					for (size_t m = 0; m < x->getKeyCount(); m++)
					{
						auto time = Real(x->getKeyTime()[m]) / REFTIME;
						auto value = x->getKeyValue()[m];
						//std::cout << "x : " << value <<std::endl;
						bone->m_Translation_Times[0].push_back(time);
						bone->m_Translation_Values[0].push_back(value);
					}
				}
				if (y) 
				{
					for (size_t m = 0; m < y->getKeyCount(); m++)
					{
						auto time = Real(y->getKeyTime()[m]) / REFTIME;
						auto value = y->getKeyValue()[m];
						//std::cout << "y : " << value << std::endl;
						bone->m_Translation_Times[1].push_back(time);
						bone->m_Translation_Values[1].push_back(value);
					}
				}

				if (z) 
				{
					for (size_t m = 0; m < z->getKeyCount(); m++)
					{
						auto time = Real(z->getKeyTime()[m]) / REFTIME;
						auto value = z->getKeyValue()[m];
						//std::cout << "z : " << value << std::endl;
						bone->m_Translation_Times[2].push_back(time);
						bone->m_Translation_Values[2].push_back(value);
					}
				}
			}

			if (propertyData == "Lcl Rotation")
			{
				auto x = node->getCurve(0);
				auto y = node->getCurve(1);
				auto z = node->getCurve(2);
				if (x) 
				{
					for (size_t m = 0; m < x->getKeyCount(); m++)
					{
						auto time = Real(x->getKeyTime()[m]) / REFTIME;
						auto value = x->getKeyValue()[m];

						bone->m_Rotation_Times[0].push_back(time);
						bone->m_Rotation_Values[0].push_back(value);
					}
				}

				if (y) 
				{
					for (size_t m = 0; m < y->getKeyCount(); m++)
					{
						auto time = Real(y->getKeyTime()[m]) / REFTIME;
						auto value = y->getKeyValue()[m];

						bone->m_Rotation_Times[1].push_back(time);
						bone->m_Rotation_Values[1].push_back(value);
					}
				}

				if (z) 
				{
					for (size_t m = 0; m < z->getKeyCount(); m++)
					{
						auto time = Real(z->getKeyTime()[m]) / REFTIME;
						auto value = z->getKeyValue()[m];

						bone->m_Rotation_Times[2].push_back(time);
						bone->m_Rotation_Values[2].push_back(value);
					}
				}

			}

			if (propertyData == "Lcl Scaling")
			{
				auto x = node->getCurve(0);
				auto y = node->getCurve(1);
				auto z = node->getCurve(2);

				if (x) 
				{
					for (size_t m = 0; m < x->getKeyCount(); m++)
					{
						auto time = Real(x->getKeyTime()[m]) / REFTIME;
						auto value = x->getKeyValue()[m];

						bone->m_Scale_Times[0].push_back(time);
						bone->m_Scale_Values[0].push_back(value);
					}
				}
				if (y) 
				{
					for (size_t m = 0; m < y->getKeyCount(); m++)
					{
						auto time = Real(y->getKeyTime()[m]) / REFTIME;
						auto value = y->getKeyValue()[m];

						bone->m_Scale_Times[1].push_back(time);
						bone->m_Scale_Values[1].push_back(value);
					}
				}

				if (z) 
				{
					for (size_t m = 0; m < z->getKeyCount(); m++)
					{
						auto time = Real(z->getKeyTime()[m]) / REFTIME;
						auto value = z->getKeyValue()[m];

						bone->m_Scale_Times[2].push_back(time);
						bone->m_Scale_Values[2].push_back(value);
					}
				}

			}
		}


		void coutName_Type(ofbx::Object::Type ty,ofbx::Object* obj) 
		{
			{
				std::cout << obj ->name << "  -  ";
				std::cout<< obj->element.getFirstProperty()->getValue().toU64() <<"\n";


				switch (ty)
				{
				case ofbx::Object::Type::ROOT:
					printf("0_ROOT\n");
					break;

				case ofbx::Object::Type::GEOMETRY:
					printf("1_GEOMETRY\n");
					break;

				case ofbx::Object::Type::SHAPE:
					printf("2_SHAPE\n");
					break;

				case ofbx::Object::Type::MATERIAL:
					printf("3_MATERIAL\n");
					break;

				case ofbx::Object::Type::MESH:
					printf("4_MESH\n");
					break;

				case ofbx::Object::Type::TEXTURE:
					printf("5_TEXTURE\n");
					break;

				case ofbx::Object::Type::LIMB_NODE:
					printf("6_LIMB_NODE\n");
					break;

				case ofbx::Object::Type::NULL_NODE:
					printf("7_NULL_NODE\n");
					break;

				case ofbx::Object::Type::NODE_ATTRIBUTE:
					printf("8_NODE_ATTRIBUTE\n");
					break;

				case ofbx::Object::Type::CLUSTER:
					printf("9_CLUSTER\n");
					break;

				case ofbx::Object::Type::SKIN:
					printf("10_SKIN\n");
					break;

				case ofbx::Object::Type::BLEND_SHAPE:
					printf("11_BLEND_SHAPE\n");
					break;

				case ofbx::Object::Type::BLEND_SHAPE_CHANNEL:
					printf("12_BLEND_SHAPE_CHANNEL\n");
					break;

				case ofbx::Object::Type::ANIMATION_STACK:
					printf("13_BLEND_SHAPE\n");
					break;

				case ofbx::Object::Type::ANIMATION_LAYER:
					printf("14_BLEND_SHAPE\n");
					break;

				case ofbx::Object::Type::ANIMATION_CURVE:
					printf("15_BLEND_SHAPE\n");
					break;

				case ofbx::Object::Type::ANIMATION_CURVE_NODE:
					printf("16_ANIMATION_CURVE_NODE\n");
					break;

				case ofbx::Object::Type::POSE:
					printf("17_POSE  ");
					break;

				default:
					break;
				}

			}
		}
	};
}