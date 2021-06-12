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

namespace dyno {
	template<typename TDataType> class TriangleSet;

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
		DECLARE_CLASS_1(VolumeGenerator, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VolumeGenerator();
		~VolumeGenerator() override;

		void load(std::string filename);
	public:
		void makeLevelSet();

		int ni;
		int nj;
		int nk;
		
		float dx;

		CArray<Vec3ui> tri;
		CArray<Vec3f> x;
		Vec3f origin;
		CArray3f phi;

		std::shared_ptr<TriangleSet<TDataType>> closedSurface;
	};
}
