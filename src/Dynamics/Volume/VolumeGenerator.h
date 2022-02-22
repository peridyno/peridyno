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
#include "Volume.h"
#include "Array/Array3D.h"

#include "Topology/TriangleSet.h"

namespace dyno {
	typedef Vector<unsigned int, 3> Vec3ui;
	typedef Vector<int, 3> Vec3i;

	typedef CArray3D<unsigned int> CArray3ui;
	typedef CArray3D<float> CArray3f;
	typedef CArray3D<int> CArray3i;

	/**
	 * @brief This is a GPU-based implementation of grid-based signed distance field 
	 *			(level set) generator for triangle meshes.
	 * 		  For more details, please refer to Robert Bridson's website (www.cs.ubc.ca/~rbridson).
	 */
	template<typename TDataType>
	class VolumeGenerator : public Volume<TDataType>
	{
		DECLARE_TCLASS(VolumeGenerator, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VolumeGenerator();
		~VolumeGenerator() override;

		void load(std::string filename);

		void updateVolume() override;

	public:
		DEF_INSTANCE_IN(TriangleSet<TDataType>, ClosedSurface, "");

		DEF_VAR_IN(Real, Spacing, "");

		DEF_VAR_IN(uint, Padding, "");

	public:
		void makeLevelSet();

		int ni;
		int nj;
		int nk;

		CArray<Vec3ui> faceList;
		CArray<Vec3f> vertList;
		Vec3f origin;
		CArray3f phi;
	};
}
