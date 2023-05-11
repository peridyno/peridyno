/**
 * Copyright 2023 Xiaowei He
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

namespace dyno
{
	template<typename TDataType>
	class MergeTriangleSet : public Node
	{
		DECLARE_TCLASS(MergeTriangleSet, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		MergeTriangleSet();
		~MergeTriangleSet() override;

		inline std::string caption() override { return "Merge"; }

	public:
		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "The merged triangle set");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, First, "The first triangle set");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, Second, "The second triangle set");

	protected:
		void resetStates() override;
	};

	IMPLEMENT_TCLASS(MergeTriangleSet, TDataType)
}