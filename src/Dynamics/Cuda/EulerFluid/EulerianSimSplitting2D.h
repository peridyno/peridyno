/**
 * Copyright 2023 Lixin Ren
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
#include "EulerianSim.h"
#include "Topology/AdaptiveGridSet2D.h"
#include "Topology/PointSet.h"

namespace dyno
{
	template<typename TDataType>
	class EulerianSimSplitting2D : public EulerianSim<TDataType>
	{
		DECLARE_TCLASS(EulerianSimSplitting2D, TDataType)

	public:
		typedef typename TDataType::Real Real;
		//typedef typename TDataType::Coord Coord;
		typedef typename Vector<Real, 2> Coord2D;
		typedef typename Vector<Real, 3> Coord;


		EulerianSimSplitting2D();
		~EulerianSimSplitting2D() override;

		void extrapolate2D();
		void advect2D(Real dt);
		void body_force2D(Real dt);
		void solve_pressure2D(DArray<Real>& pressure, DArray<Coord2D>& velocity, Real dt);
		void update_velocity2D(DArray<Coord2D>& velocity, DArray<Real>& pressure, Real dt);

		void node_topology();
		void vertex_topology();
		void particlesToGrids();
		void gridsToParticles(DArray<Coord2D>& velocity);
		void interpolate_previous(DArray<Coord2D>& position, DArray<Coord2D>& velocity);

		void resetStates() override;
		void updateStates() override;

		DEF_INSTANCE_IN(AdaptiveGridSet2D<TDataType>, AdaptiveVolume2D, "");

		DEF_VAR(Real, SandDensity, 3000.0f, "");
		DEF_VAR(Real, UpdateCoefficient, 0.3f, "");

		DEF_VAR_IN(Real, Radius, "The radius of the circle");
		DEF_VAR_IN(Coord2D, Center, "The center of the circle");
		DEF_ARRAY_STATE(Coord, PPosition, DeviceType::GPU, "the position of particles");
		DEF_ARRAY_STATE(Coord, PVelocity, DeviceType::GPU, "the velocity of particles");
		DEF_VAR(Real, SamplingDistance, 0.1, "Sampling distance of particles");

		DEF_ARRAY_OUT(Coord, NodeType, DeviceType::GPU, "");
		//DEF_INSTANCE_OUT(PointSet<TDataType>, LeafNodes, "Topology");

	private:
		DArray<AdaptiveGridNode2D> m_node;
		DArray<Real> m_sdf;
		DArrayList<int> m_neighbor;

		DArray<int> mNode2Ver;//anti-clockwise direction (-1,-1),(1,-1),(1,1),(-1,1)
		DArrayList<int> mVer2Node;

		DArray<CellType> m_identifier;
		DArray<Coord2D> m_velocity;
		DArray<Real> m_pressure;
		DArray<Real> m_density;

		//DArray<int> m_NodeIndex;
		//DArray<uint> m_NodeParticleNum;

		//DArray<uint> matrix_count;

		Real water_density = 1000.0f;

		Coord2D inlet_velocity = Coord2D(0.0f, 1.0f);
		Real convergence_standard = 0.001, convergence_indicator = 10;
		int number_iterations = 1;
	};
}