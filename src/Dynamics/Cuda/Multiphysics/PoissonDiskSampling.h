/**
 * Copyright 2022 Shusen Liu
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
#include "Node.h"

#include "Primitive/Primitive3D.h"
#include "Topology/PointSet.h"
#include "Samplers/Sampler.h"
#include "Field/FilePath.h"
#include "Topology/DistanceField3D.h"
#include <Volume/VolumeOctree.h>
#include "SdfSampler.h"
namespace dyno
{
	struct GridIndex
	{
		int i, j, k;
	};

	template<typename TDataType>
	class PoissonDiskSampling : public SdfSampler<TDataType>
	{
		DECLARE_TCLASS(PoissonDiskSampling, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		PoissonDiskSampling();

		void ConstructGrid();

		bool collisionJudge2D(Coord point);
		bool collisionJudge(Coord point);

		std::shared_ptr<DistanceField3D<TDataType>> loadSdf();

		DEF_VAR(Coord, Box_a, 0.0f, "Lower boudary of the sampling area");
		DEF_VAR(Coord, Box_b, 0.1f, "Upper boundary of the sampling area");

		//.SDF file
		DEF_VAR(FilePath, SdfFileName, "", "");

		Real lerp(Real a, Real b, Real alpha);

		Real getDistanceFromSDF(const Coord& p, Coord& normal);

		std::shared_ptr<DistanceField3D<TDataType>>  getSDF() {
			return inputSDF;
		}

		Coord getOnePointInsideSDF();

	private:

	protected:

		void resetStates() override;

		GridIndex searchGrid(Coord point);
		int indexTransform(int i, int j, int k);

		int pointNumberRecommend();

		std::vector<int> m_grid;

		int gnum;		//total number of grids
		int nx, ny, nz;		//grid number 

		Coord area_a, area_b;	//box 

		Coord seed_point;	// Initial point

		Real dx;		//resolution of grid 

		std::vector<Coord> points;

		unsigned int desired_points;

		GridIndex gridIndex;

		unsigned int attempted_Times = 10;

		std::shared_ptr<DistanceField3D<TDataType>> inputSDF;

		bool SDF_flag = false;

		DArray<Real> m_dist;


		//SDF in host.
		CArray3D<Real> host_dist;
		Real m_h;
		Coord m_left;

	};




	IMPLEMENT_TCLASS(PoissonDiskSampling, TDataType);

}