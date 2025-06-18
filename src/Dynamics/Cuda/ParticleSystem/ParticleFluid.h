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
#include "ParticleSystem.h"

#include "Emitters/ParticleEmitter.h"

#include "Topology/PointSet.h"

namespace dyno
{
	template<typename TDataType>
	class ParticleFluid : public ParticleSystem<TDataType>
	{
		DECLARE_TCLASS(ParticleFluid, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleFluid();
		~ParticleFluid() override;

		DEF_VAR(Real, SamplingDistance, 0.005, "Sampling distance");

		DEF_VAR(Real, SmoothingLength, 1.5, "The smoothing length is preferably set to a value between 1.5 and 2.5");

		DEF_VAR(bool, ReshuffleParticles, false, "");

	public:
		DEF_NODE_PORTS(ParticleEmitter<TDataType>, ParticleEmitter, "Particle Emitters");

		DEF_NODE_PORTS(ParticleSystem<TDataType>, InitialState, "Initial Fluid Particles");

		DEF_VAR_STATE(Real, SamplingDistance, 0.005, "The state of sampling distance");

		DEF_VAR_STATE(Real, SmoothingLength, 0.01, "The state of smoothing length");

	protected:
		void resetStates() override;

		void preUpdateStates() override;

	private:
		void loadInitialStates();

		void reshuffleParticles();
	};
}