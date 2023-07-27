/**
 * Copyright 2017-2022 Xiaowei He
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

#include "Ocean.h"
#include "Vessel.h"

#include "Algorithm/Reduction.h"

namespace dyno
{
	template<typename TDataType>
	class Coupling : public Node
	{
		DECLARE_TCLASS(Coupling, TDataType)
	public:
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		Coupling();
		~Coupling() override;

	public:
		DEF_VAR(Real, Dragging, Real(0.98), "");

		DEF_NODE_PORTS(Vessel<TDataType>, Vessel, "Vessel");
		DEF_NODE_PORT(Ocean<TDataType>, Ocean, "Ocean");

	protected:
		void resetStates() override;
		void updateStates() override;

	private:
		DArray<Coord> mForce;
		DArray<Coord> mTorque;

		Reduction<Coord> mReduce;
	};

	IMPLEMENT_TCLASS(Coupling, TDataType)
}
