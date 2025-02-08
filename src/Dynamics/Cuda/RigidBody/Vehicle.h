/**
 * Copyright 2024 Yuzhong Guo
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
#include "ArticulatedBody.h"

#include "STL/Pair.h"

#include "FilePath.h"
#include "Topology/EdgeSet.h"

#include "Field/VehicleInfo.h"

namespace dyno
{
	template<typename TDataType>
	class Jeep : virtual public ArticulatedBody<TDataType>
	{
		DECLARE_TCLASS(Jeep, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		Jeep();
		~Jeep() override;

	protected:
		void resetStates() override;
	};

	template<typename TDataType>
	class Tank : virtual public ArticulatedBody<TDataType>
	{
		DECLARE_TCLASS(Tank, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		Tank();
		~Tank() override;

	protected:
		void resetStates() override;

	};


	template<typename TDataType>
	class TrackedTank : virtual public ArticulatedBody<TDataType>
	{
		DECLARE_TCLASS(TrackedTank, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		TrackedTank();
		~TrackedTank() override;

		DEF_INSTANCE_STATE(EdgeSet<TDataType>, caterpillarTrack, "");
	protected:
		void resetStates() override;

	private:


	};
	

	template<typename TDataType>
	class UAV : virtual public ArticulatedBody<TDataType>
	{
		DECLARE_TCLASS(UAV, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		UAV();
		~UAV() override;

	protected:
		void resetStates() override;

	};


	template<typename TDataType>
	class UUV : virtual public ArticulatedBody<TDataType>
	{
		DECLARE_TCLASS(UUV, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		UUV();
		~UUV() override;

	protected:
		void resetStates() override;

	};

}

