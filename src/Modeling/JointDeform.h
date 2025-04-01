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
#include "ModelEditing.h"
#include "Topology/TriangleSet.h"
#include "Topology/TextureMesh.h"

#include "Module/ComputeModule.h"

#include "GLPointVisualModule.h"
#include "GLWireframeVisualModule.h"

#include "Field/FilePath.h"
#include "SkinInfo.h"
#include "JointInfo.h"

namespace dyno
{
	/**
	 * @brief A class to facilitate showing the shape information
	 */

	template<typename TDataType>
	class JointDeform : public ModelEditing<TDataType>
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
		DEF_ARRAYLIST_IN(Transform3f, InstanceTransform, DeviceType::GPU, "InstanceTransform");
		DEF_INSTANCE_STATE(TextureMesh, TextureMesh, "");

	protected:
		void resetStates() override;


		void updateStates() override;




	private:

		void updateAnimation(int frameNumber);


	};



	IMPLEMENT_TCLASS(JointDeform, TDataType);
}