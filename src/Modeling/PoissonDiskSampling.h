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

#include "Topology/Primitive3D.h"
#include "Topology/PointSet.h"
#include "SamplingPoints.h"



namespace dyno
{
	template<typename TDataType>
	class PoissonDiksSampling : public SamplingPoints<TDataType>
	{
		DECLARE_TCLASS(PoissonDiksSampling, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		PoissonDiksSampling();


		void ConstructGrid();

		bool collisionJudge(Coord point);


		DEF_VAR(Real, SamplingDistance, 0.01, "Sampling distance");

		DEF_VAR(int, Dimension, 2, "Dimensions of sampling erea ");

		DEF_VAR(int, PointsNumber, 2, "Desired samples");

	protected:
		void resetStates() override;

		void searchGrid(Coord point, int& i, int &j, int&k);

		int indexTransform(int i, int j, int k);

		std::vector<int> m_grid;

		int gnum;
		int nx, ny, nz;
 
		Coord area_a, area_b;

		Coord seed_point;

		Real dx;

		std::vector<Coord> points;

	};




	IMPLEMENT_TCLASS(PoissonDiksSampling, TDataType);

}