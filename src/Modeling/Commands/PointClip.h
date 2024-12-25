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
#include "Node/ParametricModel.h"

namespace dyno
{


	template<typename TDataType>
	class PointClip : public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(PointClip, TDataType);


	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		PointClip();



		DEF_INSTANCE_IN(PointSet<TDataType>, PointSet, "");

		DEF_VAR(Real, PlaneSize, 10, "");
		DEF_VAR(bool, Reverse, false, "");
		DEF_VAR(Real, PointSize, 0.008, "");
		DEF_VAR(Color,PointColor,Color(1,0,0),"");
		DEF_VAR(bool, ShowPlane, false, "");


		DEF_INSTANCE_STATE(TriangleSet<TDataType>,ClipPlane , "");
		DEF_INSTANCE_STATE(PointSet<TDataType>, PointSet, "");


	public:



	protected:
		void resetStates() override;
		void updateStates() override;
		void clip();
		void transformPlane();
		void showPlane();


	private:

		Vec3f Normal;
		std::vector<Coord> planeVertices;
		std::shared_ptr<GLSurfaceVisualModule> surface;
		std::shared_ptr<GLPointVisualModule> glpoint;
	};



	IMPLEMENT_TCLASS(PointClip, TDataType);
}