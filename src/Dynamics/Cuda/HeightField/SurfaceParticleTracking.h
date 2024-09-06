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

#include "GranularMedia.h"

#include "Topology/PointSet.h"

namespace dyno
{
	/**
	 * Parallel implementation of the paper "Fast Poisson Disk Sampling in Arbitrary Dimensions" to generate and track particles for granular flows
	 * For more details, refer to Sec.6 of "Shallow Sand Equations: Real-Time Height Field Simulation of Dry Granular Flows" by Zhu et al.[2021], IEEE TVCG.
	 */
	template<typename TDataType>
	class SurfaceParticleTracking : public Node
	{
		DECLARE_TCLASS(SurfaceParticleTracking, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename Vector<Real, 3> Coord3D;
		typedef typename Vector<Real, 4> Coord4D;

		SurfaceParticleTracking();
		~SurfaceParticleTracking();

	public:
		DEF_VAR(uint, Layer, 1, "");
		DEF_VAR(Real, Spacing, 0.25f, "Particle sampling distance");

		DEF_NODE_PORT(GranularMedia<TDataType>, GranularMedia, "");

	public:
		DEF_INSTANCE_STATE(PointSet<TDataType>, PointSet, "Particles used to visualize granular media");

	protected:
		void resetStates() override;

		void updateStates() override;

		bool validateInputs() override;

	private:
		void advect();

		void deposit();

		void generate();

		void updatePointSet();

	private:
		int mNx;
		int mNy;
		int mNz;

		DArray3D<Coord3D> mPosition;
		DArray3D<Coord3D> mPositionBuffer;

		DArray3D<bool> mMask;
		DArray3D<bool> mMaskBuffer;

		DArray3D<uint> mMutex;
	};
}