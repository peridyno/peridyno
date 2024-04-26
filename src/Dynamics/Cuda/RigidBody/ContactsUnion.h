/**
 * Copyright 2017-2021 Xiaowei He
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
#include "Collision/CollisionData.h"

#include "Module/ComputeModule.h"
#include "Topology/Joint.h"

namespace dyno 
{
	template<typename TDataType>
	class ContactsUnion : public ComputeModule
	{
		DECLARE_TCLASS(ContactsUnion, TDataType)
	public:
		typedef typename ::dyno::TContactPair<Real> ContactPair;
		typedef typename BallAndSocketJoint<Real> BallAndSocketJoint;
		typedef typename SliderJoint<Real> SliderJoint;
		typedef typename HingeJoint<Real> HingeJoint;
		typedef typename FixedJoint<Real> FixedJoint;

		ContactsUnion() {};
		~ContactsUnion() override {};

		void compute() override;

		void filterArray(CArray<int>& mapMatrix, CArray<ContactPair>& contactsA, CArray<ContactPair>& contactsB, CArray<ContactPair>&contactsC, int bodynum);

	public:
		DEF_ARRAY_IN(Real, Mass, DeviceType::GPU, "Mass of rigid bodies");

		DEF_ARRAY_IN(BallAndSocketJoint, BallAndSocketJoints, DeviceType::GPU, "Ball And Socket Joints");

		DEF_ARRAY_IN(SliderJoint, SliderJoints, DeviceType::GPU, "Slider Joints");

		DEF_ARRAY_IN(HingeJoint, HingeJoints, DeviceType::GPU, "Hinge Joints");

		DEF_ARRAY_IN(FixedJoint, FixedJoints, DeviceType::GPU, "Fixed Joints");

		DEF_ARRAY_IN(ContactPair, ContactsA, DeviceType::GPU, "");
		DEF_ARRAY_IN(ContactPair, ContactsB, DeviceType::GPU, "");

		DEF_ARRAY_OUT(ContactPair, Contacts, DeviceType::GPU, "");

	protected:
		bool validateInputs() override;
	};
}
