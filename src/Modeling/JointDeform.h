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
#include "Node.h"
#include "Topology/TriangleSet.h"
#include "Topology/TextureMesh.h"

#include "Module/ComputeModule.h"

#include "GLPointVisualModule.h"
#include "GLWireframeVisualModule.h"

#include "FilePath.h"
#include "SkinInfo.h"
#include "JointInfo.h"
#include "GltfFunc.h"

namespace dyno
{
	/**
	 * @brief A class to facilitate showing the shape information
	 */

	template<typename TDataType>
	class JointDeform : public Node
	{
		DECLARE_TCLASS(JointDeform, TDataType);

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

		JointDeform();
		~JointDeform();

	public:


		DEF_INSTANCE_IN(JointInfo, Joint, "Joint");
		DEF_INSTANCE_IN(SkinInfo, Skin, "Skin");	
		DEF_INSTANCE_STATE(TextureMesh, TextureMesh, "");

	protected:
		void resetStates() override
		{
			auto& joint = this->inJoint()->getData();
			auto& skin = this->inSkin()->getData();

			if (joint.isEmpty() || skin.isEmpty())
			{
				return;
			}
			
			auto textureMesh = this->stateTextureMesh()->getDataPtr();


			textureMesh->vertices() = skin.mesh->vertices();
			textureMesh->normals() = skin.mesh->normals();
			textureMesh->texCoords() = skin.mesh->texCoords();
			textureMesh->shapeIds() = skin.mesh->shapeIds();

			textureMesh->shapes() = skin.mesh->shapes();
			textureMesh->materials() = skin.mesh->materials();

		}

		void updateStates() override 
		{
			int frameNumber = this->stateFrameNumber()->getValue();


			auto& joint = this->inJoint()->getData();
			auto& skin = this->inSkin()->getData();

			if (joint.isEmpty() || skin.isEmpty()) 
			{
				return;
			}


			updateAnimation(frameNumber);
		}



	private:

		void updateAnimation(int frameNumber) 
		{
			//update Points
			auto& skinInfo = this->inSkin()->getData();
			auto& jointInfo = this->inJoint()->getData();

			auto textureMesh = this->stateTextureMesh()->getDataPtr();


			for (size_t i = 0; i < skinInfo.size(); i++)//
			{
				auto& bindJoint0 = skinInfo.V_jointID_0[i];
				auto& bindJoint1 = skinInfo.V_jointID_1[i];

				auto& bindWeight0 = skinInfo.V_jointWeight_0[i];
				auto& bindWeight1 = skinInfo.V_jointWeight_1[i];

				for (size_t j = 0; j < skinInfo.skin_VerticeRange[i].size(); j++)
				{
					//
					Vec2u& range = skinInfo.skin_VerticeRange[i][j];

					skinAnimation(skinInfo.initialPosition,
						textureMesh->vertices(),
						jointInfo.JointInverseBindMatrix,
						jointInfo.JointWorldMatrix,

						bindJoint0,
						bindJoint1,
						bindWeight0,
						bindWeight1,
						Mat4f::identityMatrix(),
						false,
						range
					);

					//update Normals

					skinAnimation(
						skinInfo.initialNormal,
						textureMesh->normals(),
						jointInfo.JointInverseBindMatrix,
						jointInfo.JointWorldMatrix,

						bindJoint0,
						bindJoint1,
						bindWeight0,
						bindWeight1,
						Mat4f::identityMatrix(),
						true,
						range
					);

				}
			}
			
			
		}

	};



	IMPLEMENT_TCLASS(JointDeform, TDataType);
}