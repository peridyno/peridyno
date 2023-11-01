/**
 * Copyright 2017-2021 Xiaowei He
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
#include "Module/VisualModule.h"

namespace dyno
{
	class VkVisualModule : public VisualModule
	{
	public:
		VkVisualModule();
		virtual ~VkVisualModule();

		std::string getModuleType() override { return "VkVisualModule"; }

		virtual void updateGraphicsContext() {};

		virtual void buildCommandBuffers(VkCommandBuffer drawCmdBuffer) = 0;
		virtual void prepare(VkRenderPass renderPass) = 0;
		virtual void viewChanged(const glm::mat4& perspective, const glm::mat4& view) = 0;

	protected:
		void updateImpl() override;
	};
}

