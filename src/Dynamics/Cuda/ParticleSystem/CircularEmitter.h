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
#include "ParticleEmitter.h"

#include "Topology/EdgeSet.h"

namespace dyno
{
	template<typename TDataType>
	class CircularEmitter : public ParticleEmitter<TDataType>
	{
		DECLARE_TCLASS(CircularEmitter, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		CircularEmitter();
		virtual ~CircularEmitter();
		
	public:
		DEF_VAR(Real, Radius, 0.1, "Emitter radius");

		DEF_INSTANCE_STATE(EdgeSet<TDataType>, Outline, "Outline of the emitter");

	private:
		void resetStates() override;

		void generateParticles() override;

		void tranformChanged();
	};

	IMPLEMENT_TCLASS(CircularEmitter, TDataType)
}