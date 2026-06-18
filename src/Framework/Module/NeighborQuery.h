/**
 * Copyright 2026 Xiaowei He
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

namespace dyno
{
	/**
	 * @brief NeighborQuery and its subclasses support two operating modes. 
	 *		In the first mode, calling update() directly executes both performBroadPhase() and performNarrowPhase() sequentially. 
	 *		In the second mode, performBroadPhase() and performNarrowPhase() can be invoked separately at different points in the code.
	 */
	class NeighborQuery : public Module
	{
	public:
		NeighborQuery();
		~NeighborQuery() override;

	public:
		void performBroadPhase();

		void performNarrowPhase();

	protected:
		void updateImpl() final;

		virtual void broadphase() {};

		virtual void narrowphase() {};
	};
}
