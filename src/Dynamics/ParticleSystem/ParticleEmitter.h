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
#include "Node.h"

namespace dyno
{
	/*!
	*	\class	ParticleEimitter
	*	\brief	
	*/
	template<typename TDataType>
	class ParticleEmitter : public Node
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleEmitter(std::string name = "particle emitter");
		virtual ~ParticleEmitter();

		virtual void generateParticles();

		uint sizeOfParticles() { return mPosition.size(); }

		DArray<Coord>& getPositions() { return mPosition; }
		DArray<Coord>& getVelocities() { return mVelocity; }

		DEF_VAR(Real, VelocityMagnitude, 1, "Emitter Velocity");
		DEF_VAR(Real, SamplingDistance, 0.005, "Emitter Sampling Distance");

	protected:
		void updateStates() final;

	protected:
		DArray<Coord> mPosition;
		DArray<Coord> mVelocity;
	};
}