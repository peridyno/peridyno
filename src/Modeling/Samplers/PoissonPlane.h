/**
 * Copyright 2023 Shusen Liu
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

#include "Topology/PointSet.h"
#include "Module/ComputeModule.h"


namespace dyno
{
	template<typename TDataType>
	class PoissonPlane : public ComputeModule
	{
		DECLARE_TCLASS(PoissonPlane, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		PoissonPlane();

		void ConstructGrid();

		bool collisionJudge(Vec2f point);

		DEF_VAR(Real, SamplingDistance, 0.015, "Sampling distance");

		DEF_VAR(Vec2f, Upper, Vec2f(0.2f, 0.2f), "");

		DEF_VAR(Vec2f, Lower, Vec2f(0.1f, 0.1f), "");

		void compute() override;

		std::vector<Vec2f> getPoints()
		{
			return points;
		}


	protected:

		Vec2u searchGrid(Vec2f point);

		int indexTransform(uint i, uint j);

		int pointNumberRecommend();

		std::vector<int> m_grid;

		Vec2f mOrigin, mUpperBound;

		int gnum;		//total number of grids

		int nx, ny, nz;		//grid number 

		float dx;		//resolution of grid 

		std::vector<Vec2f> points;

		unsigned int desired_points = 150;	//desired points number

		Vec2u gridIndex;
	};




	IMPLEMENT_TCLASS(PoissonPlane, TDataType);

}