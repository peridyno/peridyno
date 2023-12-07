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

namespace dyno {
	class GroupModule : public Module
	{
	public:
		GroupModule();
		virtual ~GroupModule();

		void pushModule(std::shared_ptr<Module> m);

		const std::list<Module*>& moduleList() const { return mModuleList; }

		void setParentNode(Node* node) override;

	protected:
		void preprocess() final;
		void updateImpl() override;

	private:
		void reconstructPipeline();

		bool mModuleUpdated = false;

		std::map<ObjectId, std::shared_ptr<Module>> mModuleMap;
		std::list<Module*>  mModuleList;

		std::list<std::shared_ptr<Module>> mPersistentModule;
	};
}

