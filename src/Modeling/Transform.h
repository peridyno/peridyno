/**
 * Copyright 2022 Shusen Liu
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
#include "GLPointVisualModule.h"


namespace dyno
{


	template<typename TDataType>
	class TransformModel : public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(TransformModel, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		enum inputType
		{
			Point_ = 0,
			Edge_ = 1,
			Triangle_ = 2,
			Null_ = 3
		};

		TransformModel();

	public:
		//DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "")
		DEF_INSTANCE_IN(TopologyModule, Topology, "")

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");
		DEF_INSTANCE_STATE(PointSet<TDataType>, PointSet, "");
		DEF_INSTANCE_STATE(EdgeSet<TDataType>, EdgeSet, "");
		
		inputType inType = inputType::Null_;
		void disableRender();
		void Transform();

	protected:
		void resetStates() override;
		
		std::shared_ptr <GLSurfaceVisualModule> glModule;
		std::shared_ptr <GLWireframeVisualModule> glWireModule;
		std::shared_ptr <GLPointVisualModule> glPointModule;

	};



	IMPLEMENT_TCLASS(TransformModel, TDataType);
}