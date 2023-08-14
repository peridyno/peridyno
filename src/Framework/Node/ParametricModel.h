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
#include "Node.h"

#include "Quat.h"

#include "Primitive/Primitive3D.h"

namespace dyno
{
	template<typename TDataType>
	class ParametricModel : public Node
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParametricModel();

		Quat<Real> computeQuaternion()
		{
			auto rot = this->varRotation()->getValue();

			Quat<Real> q =
				  Quat<Real>(Real(M_PI) * rot[2] / 180, Coord(0, 0, 1))
				* Quat<Real>(Real(M_PI) * rot[1] / 180, Coord(0, 1, 0))
				* Quat<Real>(Real(M_PI) * rot[0] / 180, Coord(1, 0, 0));
			q.normalize();
			return q;
		}

	public:
		DEF_VAR(Coord, Location, 0, "Node location");
		DEF_VAR(Coord, Rotation, 0, "Node rotation");
		DEF_VAR(Coord, Scale, Coord(1), "Node scale");
	};
}