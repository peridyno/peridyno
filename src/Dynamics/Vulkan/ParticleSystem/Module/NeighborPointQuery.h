/**
 * Copyright 2021 Xiaowei He
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
#include "Module/ComputeModule.h"

namespace dyno 
{
	class NeighborPointQuery : public ComputeModule
	{
		DECLARE_CLASS(NeighborPointQuery)
	public:
		NeighborPointQuery();
		~NeighborPointQuery() override;
		
		void compute() override;

	private:
		void requestDynamicNeighborIds();

		void requestFixedSizeNeighborIds();

	public:
		/**
		* @brief Search radius
		* A positive value representing the radius of neighborhood for each point
		*/
		DEF_VAR_IN(float, Radius, "Search radius");

		/**
		 * @brief A set of points to be required from.
		 */
		DEF_ARRAY_IN(Vec3f, Position, DeviceType::GPU, "A set of points to be required from");

		/**
		 * @brief Another set of points the algorithm will require neighbor ids for.
		 *		  If not set, the set of points in Position will be required.
		 */
		DEF_ARRAY_IN(Vec3f, Other, DeviceType::GPU,
			"Another set of points the algorithm will require neighbor ids for. If not set, the set of points in Position will be required.");

		/**
		 * @brief Ids of neighboring particles
		 */
		DEF_ARRAYLIST_OUT(uint32_t, NeighborIds, DeviceType::GPU, "Return neighbor ids");
	};
}