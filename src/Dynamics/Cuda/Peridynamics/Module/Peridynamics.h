/**
 * Copyright 2021 Xiaowei He
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
#include "Module/GroupModule.h"

#include "../Bond.h"

namespace dyno
{
	/*!
	*	\class	ParticleSystem
	*	\brief	Projective peridynamics
	*
	*	This class implements the projective peridynamics.
	*	Refer to He et al' "Projective peridynamics for modeling versatile elastoplastic materials" for details.
	*/
	template<typename TDataType>
	class Peridynamics : public GroupModule
	{
		DECLARE_TCLASS(Peridynamics, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TBond<TDataType> Bond;

		Peridynamics();
		~Peridynamics() override {};

	public:
		DEF_VAR(Real, Horizon, 0.0085, "");

		DEF_VAR_IN(Real, TimeStep, "Time step size!");

		DEF_ARRAY_IN(Coord, X, DeviceType::GPU, "");
		DEF_ARRAY_IN(Coord, Y, DeviceType::GPU, "");
		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "");
		DEF_ARRAY_IN(Coord, Force, DeviceType::GPU, "");

		DEF_ARRAYLIST_IN(Bond, Bonds, DeviceType::GPU, "Storing neighbors");

	protected:
	};
}