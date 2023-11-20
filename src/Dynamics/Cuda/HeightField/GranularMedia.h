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

#include "Topology/HeightField.h"

namespace dyno
{
	/*!
	*	\class	GranularMedia
	*	\brief	This class implements the shallow sand equation to simulate dry granular flows. 
	*			For more details, refer to "Shallow Sand Equations: Real-Time Height Field Simulation of Dry Granular Flows" by Zhu et al.[2021], IEEE TVCG.
	*/

	template<typename TDataType>
	class GranularMedia : public Node
	{
		DECLARE_TCLASS(GranularMedia, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename Vector<Real, 3> Coord3D;
		typedef typename Vector<Real, 4> Coord4D;

		GranularMedia();
		~GranularMedia();

	public:
		DEF_VAR(Coord3D, Origin, Coord3D(0), "");

		DEF_VAR(uint, Width, 64, "");
		DEF_VAR(uint, Height, 64, "");

		DEF_VAR(Real, Depth, 0.7f, "Depth of the dilute layer");
		DEF_VAR(Real, CoefficientOfDragForce, 0.7f, "The drag force coefficient");
		DEF_VAR(Real, CoefficientOfFriction, 0.95f, "The frictional coefficient exerted on the dilute layer");

		DEF_VAR(Real, Spacing, 1, "Grid spacing");

		DEF_VAR(Real, Gravity, -9.8, "");

		DEF_ARRAY2D_STATE(Real, LandScape, DeviceType::GPU, "");

		DEF_ARRAY2D_STATE(Coord4D, Grid, DeviceType::GPU, "");

		DEF_ARRAY2D_STATE(Coord4D, GridNext, DeviceType::GPU, "");

		DEF_INSTANCE_STATE(HeightField<TDataType>, HeightField, "Topology");

	protected:
		void resetStates() override;
		void updateStates() override;
	};

	IMPLEMENT_TCLASS(GranularMedia, TDataType)
}