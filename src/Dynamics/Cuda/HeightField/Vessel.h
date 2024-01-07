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
#include "RigidBody/RigidBody.h"

#include "Topology/TriangleSet.h"

#include "FilePath.h"

namespace dyno 
{
	template<typename TDataType>
	class Vessel : virtual public RigidBody<TDataType>
	{
		DECLARE_TCLASS(Vessel, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		Vessel();
		~Vessel() override;

		NBoundingBox boundingBox() override;

	public:
		DEF_VAR(Coord, BarycenterOffset, Coord(0), "The center offset defined in vessel's local frame");

		DEF_VAR(FilePath, EnvelopeName, getAssetPath() + "obj/boat_boundary.obj", "");

		DEF_VAR(FilePath, MeshName, getAssetPath() + "obj/boat_mesh.obj", "");

		DEF_VAR(Real, Density, Real(1000), "Density");

		DEF_VAR_STATE(Coord, Barycenter, Coord(0), "A vessel's barycenter, note it can be different from the Center");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, Envelope, "Envelope for the vessel");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, Mesh, "Surface mesh");

	protected:
		void resetStates() override;

		void updateStates() override;

	private:
		void transform();

		TriangleSet<TDataType> mInitialEnvelope;

		TriangleSet<TDataType> mInitialMesh;

		Coord mShapeCenter = Coord(0);
	};

	IMPLEMENT_TCLASS(Vessel, TDataType)
}
