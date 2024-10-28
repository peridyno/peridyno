/**
 * Copyright 2024 Xiaowei He
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
#include "ParticleSystem/ParticleSystem.h"

#include "Bond.h"

namespace dyno
{
	/*!
	*	\class	Peridynamics
	*	\brief	A base class for peridynamics-based computational paradigms
	*/
	template<typename TDataType>
	class Peridynamics : public ParticleSystem<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TBond<TDataType> Bond;

		Peridynamics();
		~Peridynamics() override;

		std::string getNodeType() override;

	public:
		DEF_NODE_PORTS(ParticleSystem<TDataType>, SolidParticle, "Initial solid particles");

	public:
		DEF_VAR(Real, Horizon, 0.01, "Horizon");

		DEF_VAR_STATE(Real, Horizon, Real(1), "A state field representing horizon");

		DEF_ARRAY_STATE(Coord, ReferencePosition, DeviceType::GPU, "Reference position");

		DEF_ARRAYLIST_STATE(Bond, Bonds, DeviceType::GPU, "Storing neighbors");

	protected:
		void resetStates() override;

	private:
		void loadSolidParticles();
	};
}