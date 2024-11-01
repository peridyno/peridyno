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
#include "Topology/TriangleSet.h"

#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"
#include "GLSurfaceVisualModule.h"
#include "GLInstanceVisualModule.h"


namespace dyno
{
	template<typename TDataType>
	class VectorVisualNode :public Node
	{
		DECLARE_TCLASS(VectorVisualNode, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VectorVisualNode();

		std::string getNodeType() override { return "Visualization"; }

		DECLARE_ENUM(LineMode,
			Line = 0,
			Cylnder = 1,
			Arrow = 2
			);
	public:
		DEF_VAR(Real, Length, 0.4, "LinLength");
		DEF_VAR(bool, Normalize, true, "Normalize");

		DEF_ENUM(LineMode, LineMode, LineMode::Line, "");
		DEF_VAR(Real, LineWidth, 0.04, " LineWidth");
		DEF_VAR(int, ArrowResolution, 8 , "");
		DEF_VAR(int,Debug,0,"debug");

		//In
		DEF_INSTANCE_IN(PointSet<TDataType>, PointSet, "");
		DEF_ARRAY_IN(Coord, InVector, DeviceType::GPU, "");
		DEF_ARRAY_IN(Real, Scalar, DeviceType::GPU, "");

		//Normal
		DEF_INSTANCE_STATE(EdgeSet<TDataType>, NormalSet, "");
		DEF_ARRAY_STATE(Coord, Normal, DeviceType::GPU, "");

		//Arrow
		DEF_INSTANCE_STATE(TriangleSet<TDataType>, ArrowCylinder, "");
		DEF_INSTANCE_STATE(TriangleSet<TDataType>, ArrowCone, "");
		DEF_ARRAY_STATE(Transform3f, TransformsCylinder, DeviceType::GPU, "Instance transform");
		DEF_ARRAY_STATE(Transform3f, TransformsCone, DeviceType::GPU, "Instance transform");


	protected:
		void resetStates() override;
		void updateStates() override;

	private:
		void varChanged();
		void renderChanged();


		std::shared_ptr<GLWireframeVisualModule> gledge;
		std::shared_ptr<GLInstanceVisualModule> glInstanceCylinder;
		std::shared_ptr<GLInstanceVisualModule> glInstanceCone;
		DArray<Coord> d_points;
		DArray<TopologyModule::Edge> d_edges;
		DArray<Coord> d_normalPt;
		DArray<Real> d_ConvertColor;
	};



	IMPLEMENT_TCLASS(VectorVisualNode, TDataType);
}