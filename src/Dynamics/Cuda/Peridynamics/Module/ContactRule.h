/**
 * Copyright 2021 Zixuan Lu
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
#include "Collision/CollisionDetectionBroadPhase.h"
#include "Topology/TriangleSet.h"
#include "Primitive/Primitive3D.h"
#include "Module.h"
#include "Module/ConstraintModule.h"

namespace dyno
{
	template<typename TDataType> class CollisionDetectionBroadPhase;

	template<typename TDataType>
	class ContactRule : public  ConstraintModule
	{
		DECLARE_TCLASS(ContactRule, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ContactRule();

		virtual ~ContactRule();

		void constrain() override;

		void initCCDBroadPhase(); // One should call it AT LEAST ONCE to initial data 'mBroadPhaseCD' before update the model to get inState or outState updated.
		void setContactMaxIte(int ite) {	
			this->ContactMaxIte = ite;	
		}
	public:
		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangularMesh, "");

		DEF_VAR_IN(Real, Xi, "Thickness");

		DEF_VAR_IN(Real, S, "Refactor s");

		DEF_VAR_IN(Real, Unit, "Maximum primitive size (edge length)");

		DEF_ARRAY_IN(Coord, OldPosition, DeviceType::GPU, "Old Vertex Position");

		DEF_ARRAY_IN(Coord, NewPosition, DeviceType::GPU, "New Vertex Position");

		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "Particle velocity");

		DEF_VAR_IN(Real, TimeStep, "");
		
	public:
		DEF_ARRAY_OUT(Coord, ContactForce, DeviceType::GPU, "Contact Force");
		DEF_ARRAY_OUT(Real, Weight, DeviceType::GPU, "Weight");
		Real weight;
	private:
		std::shared_ptr<CollisionDetectionBroadPhase<TDataType>> mBroadPhaseCD;
		DArrayList<Coord> firstTri;
		DArrayList<Coord> secondTri;
		DArrayList<int> trueContact;
		DArray<uint> trueContactCnt;
		DArray<Real> Weight;
		int ContactMaxIte = 150;
	};
}