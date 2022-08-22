/**
 * Copyright 2022 Xiaowei He
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
#include "ParametricModel.h"
#include "GLSurfaceVisualModule.h"
namespace dyno
{
	template<typename TDataType>
	class CubeModel : public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(CubeModel, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		CubeModel();

	public:
		DEF_VAR(Vec3f, Length, Real(1), "Edge length");

		DEF_VAR(Vec3i, Segments, Vec3i(1, 1, 1), "");

		DEF_INSTANCE_STATE(QuadSet<TDataType>, QuadSet, "");

		DEF_VAR_OUT(TOrientedBox3D<Real>, Cube,  "");

	protected:
		std::shared_ptr <GLSurfaceVisualModule> glModule;

		void resetStates() override;
	};

	IMPLEMENT_TCLASS(CubeModel, TDataType);
}