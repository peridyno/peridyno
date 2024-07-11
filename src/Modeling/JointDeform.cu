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

#include "GltfFunc.h"

namespace dyno
{
	/**
	 * @brief A class to facilitate showing the shape information
	 */

	template<typename TDataType>
	JointDeform<TDataType>::JointDeform()
	{
		this->stateTextureMesh()->setDataPtr(std::make_shared<TextureMesh>());

		auto render = std::make_shared<GLPhotorealisticRender>();
		this->stateTextureMesh()->connect(render->inTextureMesh());
		this->graphicsPipeline()->pushModule(render);

	}

	template<typename TDataType>
	JointDeform<TDataType>::~JointDeform() 
	{
	
	}

	DEFINE_CLASS(JointDeform);
}