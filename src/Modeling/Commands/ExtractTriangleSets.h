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
#include "Node/ParametricModel.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "Topology/TriangleSets.h"


namespace dyno
{
	template<typename TDataType>
	class ExtractTriangleSets : public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(ExtractTriangleSets, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ExtractTriangleSets();

	public:

		DEF_VAR(std::vector<int>, ID, std::vector<int>{0},"");

		DEF_VAR(std::vector<Transform3f>, ShapeTransform, std::vector<Transform3f>{}, "");

		DEF_INSTANCE_IN(TriangleSets<TDataType>, TriangleSets, "");

		DEF_INSTANCE_STATE(TriangleSets<TDataType>, TriangleSets, "");

	protected:
		void resetStates() override;

	private:

		std::vector<std::shared_ptr<TriangleSet<TDataType>>> Extract(std::shared_ptr<TriangleSets<TDataType>> triSets, std::shared_ptr<TriangleSets<TDataType>> outTriSet, std::vector<int> triSetId);

	};
	IMPLEMENT_TCLASS(ExtractTriangleSets, TDataType);
}