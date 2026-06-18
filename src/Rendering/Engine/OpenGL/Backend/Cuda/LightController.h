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
#include "STL/Pair.h"
#include "Module/KeyboardInputModule.h"

namespace dyno 
{

	class LightController : public KeyboardInputModule
	{
		DECLARE_CLASS(LightController);
	public:
		typedef typename ::dyno::Pair<uint, uint> BindingPair;

		LightController();
		~LightController() override {};

		DEF_VAR(int, VehicleID, 0, "");
		DEF_VAR(Vec3f, LightDirection,Vec3f(-1,0,0), "LightDirection");
		DEF_VAR_OUT(Vec3f, LightDirection,"LightDirection");
		DEF_ARRAYLIST_IN(uint, ShapeVehicleID, DeviceType::GPU, "ShapeVehicleID");
		DEF_ARRAYLIST_OUT(float, HeadLight, DeviceType::GPU, "");
		DEF_ARRAYLIST_OUT(float, BrakeLight, DeviceType::GPU, "");
		DEF_ARRAYLIST_OUT(float, TurnSignal, DeviceType::GPU, "");

	protected:
		void onEvent(PKeyboardEvent event) override;

	private:
		int speed = 0;
		bool active = false;
		Real angle = 0.0f;
	};
}
