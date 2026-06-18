/**
 * Copyright 2025 Lixin Ren
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
#include "Node/ParametricModel.h"
#include "Primitive/Primitive2D.h"

namespace dyno
{
	enum BasicShapeType2D
	{
		CIRCLE,
		RECTANGLE,
		UNKNOWN2D
	};

	template<typename TDataType>
	class BasicShape2D : public ParametricModel<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename dyno::Vector<Real, 2> Coord2D;

		BasicShape2D();

		std::string getNodeType() override { return "Basic Shapes 2D"; }

		virtual BasicShapeType2D getShapeType() {
			return BasicShapeType2D::UNKNOWN2D;
		};

		Coord2D computeRotate(Coord2D v)
		{
			auto rot = this->varRotation2D()->getValue();
			Real angle = Real(M_PI) * rot / 180;

			//rotation matrix: cos() sin()
			//                               * v
			//				  -sin() con()  

			Real s = glm::sin(angle);
			Real c = glm::cos(angle);
			return Coord2D(v[0] * c + v[1] * s, v[0] * (-s) + v[1] * c);
		}

		DEF_VAR(Coord2D, Location2D, 0, "Node2D location");
		DEF_VAR(Real, Rotation2D, 0, "Node2D rotation");
		DEF_VAR(Coord2D, Scale2D, Coord2D(1), "Node2D scale");

	};
}