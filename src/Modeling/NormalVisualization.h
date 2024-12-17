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
	class NormalVisualization : public Node
	{
		DECLARE_TCLASS(Normal, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		NormalVisualization();
		~NormalVisualization()
		{
			d_points.clear();
			d_edges.clear();
			d_normalPt.clear();
			d_normal.clear();
			triangleCenter.clear();
		}

		std::string getNodeType() override { return "Visualization"; }

	public:
		DEF_VAR(Real, Length, 0.2, "LinLength");
		DEF_VAR(bool, Normalize, true, "Normalize");

		DEF_VAR(Real, LineWidth, 0.01, " LineWidth");
		DEF_VAR(bool, ShowWireframe, true, "ShowWireframe");
		DEF_VAR(int, ArrowResolution, 8 , "");

		//In
		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "");
		DEF_ARRAY_IN(Coord, InNormal, DeviceType::GPU, "");
		DEF_ARRAY_IN(Real, Scalar, DeviceType::GPU, "");

		//Normal
		DEF_INSTANCE_STATE(EdgeSet<TDataType>, NormalSet, "");
		DEF_ARRAY_STATE(Coord, Normal, DeviceType::GPU, "");
		DEF_INSTANCE_STATE(PointSet<TDataType>, TriangleCenter, "");

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

		std::shared_ptr<GLInstanceVisualModule> glInstanceCylinder;
		std::shared_ptr<GLInstanceVisualModule> glInstanceCone;
		DArray<Coord> d_points;
		DArray<TopologyModule::Edge> d_edges;
		DArray<Coord> d_normalPt;
		DArray<Coord> d_normal;
		DArray<Coord> triangleCenter;
	};

	IMPLEMENT_TCLASS(NormalVisualization, TDataType);
}