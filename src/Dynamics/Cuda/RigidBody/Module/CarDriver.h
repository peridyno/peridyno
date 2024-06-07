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
#include "Module/KeyboardInputModule.h"
#include "Topology/DiscreteElements.h"

namespace dyno 
{
	template<typename TDataType>
	class CarDriver : public KeyboardInputModule
	{
		DECLARE_TCLASS(CarDriver, TDataType);
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename BallAndSocketJoint<Real> BallAndSocketJoint;
		typedef typename SliderJoint<Real> SliderJoint;
		typedef typename HingeJoint<Real> HingeJoint;
		typedef typename FixedJoint<Real> FixedJoint;
		typedef typename PointJoint<Real> PointJoint;
		typedef typename dyno::Quat<Real> TQuat;

		CarDriver();
		~CarDriver() override {};

		DEF_VAR(int,BodyId,0,"Body ID");

		DEF_VAR(std::vector<int>, DrivingWheel, std::vector<int>{0}, "");
		DEF_VAR(std::vector<int>, SteeringWheel, std::vector<int>{0},"");

		DEF_ARRAY_IN(Quat<Real>, Quaternion, DeviceType::GPU, "Quaternion");
		DEF_INSTANCE_IN(DiscreteElements<TDataType>, Topology, "Topology");


	protected:
		void onEvent(PKeyboardEvent event) override;
	};
}
