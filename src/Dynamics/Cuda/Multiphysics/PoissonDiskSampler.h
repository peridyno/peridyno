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
#include "SdfSampler.h"
namespace dyno
{
	struct GridIndex
	{
		int i, j, k;
	};

	template<typename TDataType>
	class PoissonDiskSampler : public SdfSampler<TDataType>
	{
		DECLARE_TCLASS(PoissonDiskSampler, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		PoissonDiskSampler();
		~PoissonDiskSampler();

		void resetStates() override;

	private:

		void ConstructGrid();

		bool collisionJudge2D(Coord point);

		bool collisionJudge(Coord point);

		Real lerp(Real a, Real b, Real alpha);

		Real getDistanceFromSDF(const Coord& p, Coord& normal);

		GridIndex searchGrid(Coord point);

		int indexTransform(int i, int j, int k);

		int pointNumberRecommend();

		Coord getOnePointInsideSDF();

		std::vector<int> m_grid;

		std::vector<Coord> m_points;

		CArray3D<Real> host_dist;

		int nx, ny, nz;		//grid number 

		unsigned int m_attempted_times = 10;

		Real m_grid_dx;		//resolution of grid 

		Coord area_a, area_b;	//box 

		std::shared_ptr<DistanceField3D<TDataType>> m_inputSDF;

	};

	IMPLEMENT_TCLASS(PoissonDiskSampler, TDataType);
}