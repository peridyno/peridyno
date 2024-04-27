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
#include "Module/ComputeModule.h"

#include "Topology/DiscreteElements.h"
#include "Topology/TriangleSet.h"
#include "Topology/LinearBVH.h"

#include "CollisionData.h"
#include "CollisionDetectionBroadPhase.h"

namespace dyno {
	/**
	 * A class to detect contacts between discrete elements and a triangle mesh
	 */
	template<typename TDataType>
	class CollistionDetectionTriangleSet : public ComputeModule
	{
		DECLARE_TCLASS(CollistionDetectionTriangleSet, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename ::dyno::TAlignedBox3D<Real> AABB;
		typedef typename ::dyno::TContactPair<Real> ContactPair;

		CollistionDetectionTriangleSet();
		~CollistionDetectionTriangleSet() override;

	public:
		DEF_INSTANCE_IN(DiscreteElements<TDataType>, DiscreteElements, "");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "");

		DEF_ARRAY_OUT(ContactPair, Contacts, DeviceType::GPU, "");

	protected:
		void compute() override;

	private:
		Scan<int> mScan;
		Reduction<int> mReduce;

		DArray<AABB> mQueryAABB;
		DArray<AABB> mTriangleAABB;
		LinearBVH<TDataType> mLBVH;

		DArray<int> mBoundaryContactCounter;
		DArray<int> mBoundaryContactCpy;
		DArray<ContactPair> mContactBuffer;

		std::shared_ptr<CollisionDetectionBroadPhase<TDataType>> mBroadPhaseCD;
	};

	IMPLEMENT_TCLASS(CollistionDetectionTriangleSet, TDataType)
}