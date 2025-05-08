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
#include "Topology/HierarchicalModel.h"
#include "Field/VehicleInfo.h"

namespace dyno 
{



	template<typename TDataType>
	class AnimationDriver : public KeyboardInputModule
	{
		DECLARE_TCLASS(AnimationDriver, TDataType);
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename ::dyno::BallAndSocketJoint<Real> BallAndSocketJoint;
		typedef typename ::dyno::SliderJoint<Real> SliderJoint;
		typedef typename ::dyno::HingeJoint<Real> HingeJoint;
		typedef typename ::dyno::FixedJoint<Real> FixedJoint;
		typedef typename ::dyno::PointJoint<Real> PointJoint;
		typedef typename dyno::Quat<Real> TQuat;

		AnimationDriver();
		~AnimationDriver() override {};



		DEF_VAR(Real, Speed,4,"Speed");

		DEF_VAR(std::vector<Animation2JointConfig>, BindingConfiguration, std::vector<Animation2JointConfig>(), "Animation Joint Config");

		DEF_INSTANCE_IN(DiscreteElements<TDataType>, Topology, "Topology");
		DEF_INSTANCE_IN(JointAnimationInfo, JointAnimationInfo,"Animation objects");
		DEF_VAR_IN(Real, DeltaTime,"");


	public:



	protected:

		void onEvent(PKeyboardEvent event) override;

		void getFrameAndWeight(float time,Vec3i& keyFrame,Vec3f& weight, std::vector<std::vector<Real>*> animTime)
		{
			for (size_t channel = 0; channel < 3; channel++)
			{
				float tempTime = std::fmod(time, (*animTime[channel]).back());

				std::vector<Real>& animData = *animTime[channel];
				for (size_t j = 0; j < animData.size() - 1; j++)
				{
					if (tempTime > animData[j] && tempTime <= animData[j + 1])
					{
						float v = (tempTime - animData[j]) / (animData[j + 1] - animData[j]);
						weight[channel] = v;
						keyFrame[channel] = j;
						break;
					}

				}
			}
			

		}
		


	private:

		float move = 0;
	};
}
