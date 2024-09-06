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
#include "ParticleEmitter.h"

#include "Topology/PointSet.h"

namespace dyno
{
	class ParticleFluid : public ParticleSystem
	{
		DECLARE_CLASS(ParticleFluid)
	public:
		ParticleFluid();
		~ParticleFluid() override;


		DEF_NODE_PORTS(ParticleEmitter, ParticleEmitter, "Particle Emitters");

		DEF_NODE_PORTS(ParticleSystem, InitialState, "Initial Fluid Particles");

	protected:
		void resetStates() override;

		void preUpdateStates() override;

		void postUpdateStates() override;

	private:
		void loadInitialStates();
	};
}