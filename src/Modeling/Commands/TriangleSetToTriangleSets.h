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
	class TriangleSetToTriangleSets : public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(TriangleSetToTriangleSets, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		TriangleSetToTriangleSets();



	public:

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "");

		DEF_INSTANCE_STATE(TriangleSets<TDataType>, TriangleSets, "");


	protected:
		void resetStates() override;


	private:

		std::vector<std::vector<int>> groupTrianglesByConnectivity(std::shared_ptr<TriangleSet<TDataType>> triSet);


	};
	IMPLEMENT_TCLASS(TriangleSetToTriangleSets, TDataType);
}