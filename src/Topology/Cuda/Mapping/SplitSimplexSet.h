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
	class SplitSimplexSet : public Node
	{
		DECLARE_TCLASS(SplitSimplexSet, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SplitSimplexSet();
		~SplitSimplexSet() override;

	public:
		DEF_INSTANCE_IN(SimplexSet<TDataType>, SimplexSet, "Input simplices");

		DEF_INSTANCE_OUT(EdgeSet<TDataType>, EdgeSet, "Out edges");

		DEF_INSTANCE_OUT(TriangleSet<TDataType>, TriangleSet, "");

		DEF_INSTANCE_OUT(TetrahedronSet<TDataType>, TetrahedronSet, "");


	protected:
		void resetStates() override;
	};

	IMPLEMENT_TCLASS(SplitSimplexSet, TDataType)
}