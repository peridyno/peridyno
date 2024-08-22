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
#include "Topology/TextureMesh.h"


namespace dyno
{
	/**
	 * @brief A class to merge TextureMeshs.
	 */

	template<typename TDataType>
	class TextureMeshMerge : public Node
	{
		DECLARE_TCLASS(TextureMeshMerge, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename TopologyModule::Triangle Triangle;

		TextureMeshMerge();

		~TextureMeshMerge();

	public:

		DEF_INSTANCE_IN(TextureMesh, First, "TextureMesh01");
	
		DEF_INSTANCE_IN(TextureMesh, Second, "TextureMesh02");

		DEF_INSTANCE_STATE(TextureMesh, TextureMesh, "");


	protected:
		void resetStates() override;

		void merge(const std::shared_ptr<TextureMesh>& texMesh01,const std::shared_ptr<TextureMesh>& texMesh02, std::shared_ptr<TextureMesh>& out );

	private:


	};



	IMPLEMENT_TCLASS(TextureMeshMerge, TDataType);
}