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


namespace dyno
{


	template<typename TDataType>
	class Merge : public Node
	{
		DECLARE_TCLASS(Merge, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		Merge();

	public:

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet01, "");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet02, "");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet03, "");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet04, "");

		DECLARE_ENUM(UpdateMode,
		Reset = 0,
			Tick = 1);

		DEF_ENUM(UpdateMode, UpdateMode, UpdateMode::Reset, "");

		//DEF_VAR_OUT(TCylinder3D<Real>, Cylinder, "");

		void disableRender();
		void preUpdateStates()override;
		void MergeCPU();
		void MergeGPU();

	protected:
		void resetStates() override;

		std::shared_ptr <GLSurfaceVisualModule> glModule;
	};



	IMPLEMENT_TCLASS(Merge, TDataType);
}