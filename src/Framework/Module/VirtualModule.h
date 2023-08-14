/**
 * Copyright 2021 XunKun Luo
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
    // Make a field mapping from NodeFlowScene to ModelFlowScene
    template<typename TDataType>
    class VirtualModule : public Module
	{
        DECLARE_TCLASS(VirtualModule, TDataType)
	public:
        VirtualModule();
		virtual ~VirtualModule();

		std::string getModuleType() override { return "VirtualModule"; }
	};
}