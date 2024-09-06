/**
 * Copyright 2022 Lixin Ren
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
#include "VolumeOctree.h"
#include "VolumeOctreeGenerator.h"

#include "Module/TopologyModule.h"
#include "Topology/TriangleSet.h"
#include "Primitive/Primitive3D.h"
#include "Vector.h"
#include "VolumeHelper.h"

namespace dyno {

	template<typename TDataType>
	class VolumeOctreeBoolean : public VolumeOctree<TDataType>
	{
		DECLARE_TCLASS(VolumeOctreeBoolean, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		DECLARE_ENUM(BooleanOperation,
			UNION_SET = 0,
			INTERSECTION_SET = 1,
			SUBTRACTION_SET = 2);

		VolumeOctreeBoolean();
		~VolumeOctreeBoolean() override;

		DEF_NODE_PORT(VolumeOctree<TDataType>, OctreeA, "Volume Octree A");
		DEF_NODE_PORT(VolumeOctree<TDataType>, OctreeB, "Volume Octree B");

		DEF_VAR(bool, MinDx, true, "");

		DEF_ENUM(BooleanOperation, BooleanType,BooleanOperation::INTERSECTION_SET, "Boolean operation type");

		//void setSignOperation(VolumeHelper<TDataType>::BooleanOperation msign) { this->varBooleanType()->getDataPtr()->setCurrentKey(msign); }

	protected:
		Coord lowerBound() override { return m_origin; }
		Coord upperBound() override { return m_origin + m_dx * Coord(m_nx, m_ny, m_nz); }

		bool validateInputs() override;

		void resetStates() override;

		void updateStates() override;

		void initParameter();

		void updateSignOperation();

	private:
		//原来的origin相对于新的origin的偏移量
		int m_offset_ax, m_offset_ay, m_offset_az;
		int m_offset_bx, m_offset_by, m_offset_bz;

		int m_nx;
		int m_ny;
		int m_nz;
		Real m_dx;
		Coord m_origin;
		int m_level0 = 0;

		int m_reconstructed_model = EMPTY; //0表示不需要重构，1表示重构模型A，2表示重构模型B
	};
}
