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

#include "JointDeform.h"
#include <GLPhotorealisticRender.h>

#include <GLPhotorealisticInstanceRender.h>

#include "GltfFunc.h"

namespace dyno
{
	/**
	 * @brief A class to facilitate showing the shape information
	 */

	template<typename TDataType>
	JointDeform<TDataType>::JointDeform()
	{
		//this->stateTextureMesh()->setDataPtr(std::make_shared<TextureMesh>());

		auto render = std::make_shared<GLPhotorealisticRender>();
		//this->stateTextureMesh()->connect(render->inTextureMesh());
		this->graphicsPipeline()->pushModule(render);

		auto instanceRender = std::make_shared<GLPhotorealisticInstanceRender>();
		//this->stateTextureMesh()->connect(instanceRender->inTextureMesh());
		this->inInstanceTransform()->connect(instanceRender->inTransform());
		this->graphicsPipeline()->pushModule(instanceRender);
		

	}

	template<typename TDataType>
	JointDeform<TDataType>::~JointDeform() 
	{
	
	}

	template<typename TDataType>
	void JointDeform<TDataType>::resetStates()
	{
		/*{
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

		}*/
	}

	template<typename TDataType>
	void JointDeform<TDataType>::updateStates()
	{
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
	}

	template<typename TDataType>
	void JointDeform<TDataType>::updateAnimation(float time)
	{
		{
			//update Points
			auto& skinInfo = this->inSkin()->getData();
			auto& jointInfo = this->inJoint()->getData();

			//auto textureMesh = this->stateTextureMesh()->getDataPtr();
			auto textureMesh = this->inTextureMesh()->getDataPtr();

			for (size_t i = 0; i < skinInfo.size(); i++)//
			{
				auto& bindJoint0 = skinInfo.V_jointID_0[i];
				auto& bindJoint1 = skinInfo.V_jointID_1[i];

				auto& bindWeight0 = skinInfo.V_jointWeight_0[i];
				auto& bindWeight1 = skinInfo.V_jointWeight_1[i];


					//
					Vec2u& range = skinInfo.skin_VerticeRange[i];

					skinAnimation(skinInfo.initialPosition,
						textureMesh->geometry()->vertices(),
						jointInfo.mJointInverseBindMatrix,
						jointInfo.mJointWorldMatrix,

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
						textureMesh->geometry()->normals(),
						jointInfo.mJointInverseBindMatrix,
						jointInfo.mJointWorldMatrix,

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

	DEFINE_CLASS(JointDeform);
}