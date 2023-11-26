/**
 * Copyright 2017-2023 Xiaowei He
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
#include "TopologyConstants.h"
#include "Module/TopologyModule.h"

namespace dyno
{
	/**
	 * @brief A PointSet stores the coordinates for a set of independent points
	 * 
	 * @tparam TDataType represents the template argument, which can either be set as DataType3f or DataType3d
	 */

	template<typename TDataType>
	class PointSet : public TopologyModule
	{
		DECLARE_TCLASS(PointSet, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		PointSet();
		~PointSet() override;

		void copyFrom(PointSet<TDataType>& pointSet);

		void setPoints(std::vector<Coord>& pos);
		void setPoints(DArray<Coord>& pos);
		void setSize(int size);
		

		int getPointSize() { return mCoords.size(); };

		/**
		 * @brief Return the lower and upper bounds for all points
		 */
		void requestBoundingBox(Coord& lo, Coord& hi);
		
		//Transform
		void scale(const Real s);
		void scale(const Coord s);
		void translate(const Coord t);

		virtual void rotate(const Coord angle);
		virtual void rotate(const Quat<Real> q);

		void loadObjFile(std::string filename);

		virtual bool isEmpty();

		void clear();

		/**
		 * @brief Return the array of points
		 */
		DArray<Coord>& getPoints() { return mCoords; }

	protected:
		DArray<Coord> mCoords;
	};
}

