/**
 * Copyright 2022 Xiaowei He
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
#include "Volume/VolumeOctree.h"

namespace dyno
{
	template<typename TDataType>
	class SparseVolumeClipper : public Node
	{
		DECLARE_TCLASS(SparseVolumeClipper, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SparseVolumeClipper();
		~SparseVolumeClipper() override;

	public:
		DEF_VAR(Coord, Translation, Coord(0), "");
		DEF_VAR(Coord, Rotation, Coord(0), "");

		DEF_ARRAY_STATE(Real, Field, DeviceType::GPU, "Signed distance field defined on trianglular vertices");

		DEF_ARRAY_STATE(Coord, Vertices, DeviceType::GPU, "");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");

		DEF_NODE_PORT(VolumeOctree<TDataType>, SparseVolume, "The value of SDFOctree");

	protected:
		void resetStates() override;

		void updateStates() override;
	};

	IMPLEMENT_TCLASS(SparseVolumeClipper, TDataType)
}