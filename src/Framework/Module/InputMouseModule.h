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
#include "Module.h"

#include "Topology/Primitive3D.h"

#include <queue>

namespace dyno
{
	enum PButtonType
	{
		BT_LEFT = 0,
		BT_RIGHT = 1,
		BT_MIDDLE = 2,
	};

	enum PActionType
	{
		AT_RELEASE = 0,
		AT_PRESS = 1,
		AT_REPEAT = 2
	};

	struct PMouseEvent 
	{
		PButtonType buttonType;

		PActionType actionType;

		TRay3D<float> ray;

		float x;
		float y;
	};

	class InputMouseModule : public Module
	{
	public:
		InputMouseModule();
		virtual ~InputMouseModule();

		std::string getModuleType() final { return "InputMouseModule"; }

		void enqueueEvent(PMouseEvent event);

	protected:
		virtual void onEvent(PMouseEvent event) {};

		void updateImpl() final;

		std::queue<PMouseEvent> mEventQueue;
	};
}
