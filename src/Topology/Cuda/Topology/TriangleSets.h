/**
 * Copyright 2025 Xiaowei He
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
#include "TriangleSet.h"

namespace dyno
{
	template<typename TDataType>
	class TriangleSets : public TriangleSet<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		TriangleSets();
		~TriangleSets() override;

		const uint shapeSize() { return mShapeSize; }

		uint setShapeSize(uint size) { mShapeSize = size; return mShapeSize; }

		DArray<uint>& shapeIds() { return mShapeIds; }

		void load(std::vector<std::shared_ptr<TriangleSet<TDataType>>>& tsArray);

		void appendShape(std::vector<Vec3f>& vertices, std::vector<TopologyModule::Triangle>& triangles);
		void appendShape(std::vector<Vec3f>& vertices, CArray<TopologyModule::Triangle>& triangles);
		void appendShape(DArray<Vec3f>& vertices, DArray<TopologyModule::Triangle>& triangles);


	private:
		uint mShapeSize = 0;

		DArray<uint> mShapeIds;
	};
}

