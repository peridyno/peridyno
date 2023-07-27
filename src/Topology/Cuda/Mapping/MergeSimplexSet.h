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

#include "Topology/SimplexSet.h"

#include "Topology/EdgeSet.h"
#include "Topology/TriangleSet.h"
#include "Topology/TetrahedronSet.h"

namespace dyno
{
	template<typename TDataType>
	class MergeSimplexSet : public Node
	{
		DECLARE_TCLASS(MergeSimplexSet, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		MergeSimplexSet();
		~MergeSimplexSet() override;

	public:

		DEF_INSTANCE_IN(EdgeSet<TDataType>, EdgeSet, "Input EdgeSet");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "Input TriangleSet");

		DEF_INSTANCE_IN(TetrahedronSet<TDataType>, TetrahedronSet, "Input TetrahedronSet");

		DEF_INSTANCE_OUT(SimplexSet<TDataType>, SimplexSet, "Output simplices");

	protected:
		void resetStates() override;
	};

	IMPLEMENT_TCLASS(MergeSimplexSet, TDataType)
}