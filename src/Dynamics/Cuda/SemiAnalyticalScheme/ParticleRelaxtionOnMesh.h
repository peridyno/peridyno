/**
 * Copyright 2024 Shusen Liu
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

#include "Topology/PointSet.h"
#include "ParticleSystem/ParticleSystem.h"
#include "ParticleSystem/ParticleEmitter.h"
#include "./ParticleSystem/Module/IterativeDensitySolver.h"
#include "./ParticleSystem/Module/ImplicitViscosity.h"
#include "./ParticleSystem/Module/ParticleIntegrator.h"
#include "./ParticleSystem/Module/NormalForce.h"
#include "Collision/NeighborTriangleQuery.h"
#include "TriangularMeshConstraint.h"
#include "Collision/NeighborPointQuery.h"
#include "Topology/TriangleSet.h"
#include "../../../Modeling/PointsBehindMesh.h"

namespace dyno
{

	/*
	*	@brief£ºSlightly shift the particles benhind triangle meshes to obtain more regular distribution.
	*
	*/


	template<typename TDataType>
	class ParticleRelaxtionOnMesh : public ParticleSystem<TDataType>
	{
		DECLARE_TCLASS(ParticleRelaxtionOnMesh, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleRelaxtionOnMesh();
		~ParticleRelaxtionOnMesh() override;

		//DEF_NODE_PORTS(ParticleSystem<TDataType>, InitialState, "Initial Fluid Particles");

		//DEF_NODE_PORT(PointsBehindMesh<TDataType>, PointsBehindMesh, "Input PointsBehindMesh Node");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "");

		DEF_INSTANCE_IN(PointSet<TDataType>, PointSet, "");

		DEF_ARRAY_IN(Coord, ParticleNormal, DeviceType::GPU, "");

		DEF_ARRAY_IN(int, ParticleBelongTriangleIndex, DeviceType::GPU, "");

		DEF_VAR(int, IterationNumber, 30, "");

		DEF_VAR(Real, NormalForceStrength, 0.05, "");

		DEF_VAR(Real, MeshCollisionThickness, 0.003, "");

		DEF_VAR(Real, PointNeighborLength, 0.006, "");

		DEF_VAR(Real, MeshNeighborLength, 0.012, "");

		DEF_VAR(Real, ViscosityStrength, 50.0f, "");

		DEF_VAR_STATE(Real, Delta, 0.03, "");

		DEF_VAR(int, DensityIteration, 5, "");

	protected:
		void resetStates() override;

		void preUpdateStates();

		void particleRelaxion();

	private:
		void loadInitialStates();



		std::shared_ptr<ParticleIntegrator<TDataType>> ptr_integrator;
		std::shared_ptr<NeighborPointQuery<TDataType>> ptr_nbrQuery;
		std::shared_ptr<IterativeDensitySolver<TDataType>> ptr_density;
		std::shared_ptr<ImplicitViscosity<TDataType>> ptr_viscosity;
		std::shared_ptr<NeighborTriangleQuery<TDataType>> ptr_nbrQueryTri;
		std::shared_ptr<TriangularMeshConstraint<TDataType>> ptr_meshCollision;
		std::shared_ptr<NormalForce<TDataType>> ptr_normalForce;
		
	};

	
}