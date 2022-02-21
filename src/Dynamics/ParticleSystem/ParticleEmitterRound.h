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
#include "ParticleSystem.h"
#include "ParticleEmitter.h"

namespace dyno
{
	template<typename TDataType>
	class ParticleEmitterRound : public ParticleEmitter<TDataType>
	{
		DECLARE_TCLASS(ParticleEmitterRound, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleEmitterRound(std::string name = "particleEmitter");
		virtual ~ParticleEmitterRound();

		void generateParticles() override;
		
	public:
		DEF_VAR(Real, Radius, 0.05, "Emitter radius");
	};
}