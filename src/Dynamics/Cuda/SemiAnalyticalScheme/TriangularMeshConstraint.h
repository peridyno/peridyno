/**
 * Copyright 2021 Yue Chang
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
#include "Module/ConstraintModule.h"

#include "Topology/TriangleSet.h"

namespace dyno
{
	template<typename TDataType>
	class TriangularMeshConstraint : public ConstraintModule
	{
		DECLARE_TCLASS(TriangularMeshConstraint, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename TopologyModule::Triangle Triangle;

		TriangularMeshConstraint();
		~TriangularMeshConstraint() override;


	public:
		DEF_VAR(Real, Thickness, 0.0065, "Threshold for collision detection");

		DEF_VAR(Real, TangentialFriction, 0, "Tangential friction");
		DEF_VAR(Real, NormalFriction, 0, "Normal friction");

		// Assumed particle mass for impulse computation
		DEF_VAR(Real, ParticleMass, Real(0.001), "Assumed particle mass for impulse computation");

		// Fake rigid properties for the triangle mesh to convert impulses into motion increments
		DEF_VAR(Real, MeshMass, Real(1.0), "Fake mass of the triangular mesh for 6D motion");
		DEF_VAR(Matrix, MeshInertia, Matrix::identityMatrix(), "Fake inertia matrix of the triangular mesh for 6D motion");
		DEF_VAR(Coord, MeshCenter, Coord(0), "Reference center of the triangular mesh for torque computation");

	public:
		DEF_VAR_IN(Real, TimeStep, "Time Step");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "Particle position");
		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "Particle velocity");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "");

		DEF_ARRAYLIST_IN(int, TriangleNeighborIds, DeviceType::GPU, "triangle neighbors");


	protected:
		void constrain() override;

	private:
		DArray<Coord> mPosBuffer;
		DArray<Coord> LinearImpulse;
		DArray<Coord> AngularImpulse;
		DArray<Coord> DeltaTranslation;
		DArray<Coord> DeltaRotation;


		DArray<Coord> mPreviousPosition;
		DArray<Coord> mPrivousVertex;
	};
}
