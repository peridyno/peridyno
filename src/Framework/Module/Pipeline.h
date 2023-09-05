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
#include "Module.h"

namespace dyno
{
	class Node;

	class Pipeline : public Module
	{
	public:
		Pipeline(Node* node);
		virtual ~Pipeline();

		uint sizeOfDynamicModules();
		uint sizeOfPersistentModules();

		void pushModule(std::shared_ptr<Module> m);

		void popModule(std::shared_ptr<Module> m);

		template<class TModule>
		std::shared_ptr<TModule> createModule()
		{
			std::shared_ptr<TModule> m = std::make_shared<TModule>();
			this->pushModule(m);

			return m;
		}

		/**
		 * Return the first module with TModule type
		 */
		template<class TModule>
		std::shared_ptr<TModule> findFirstModule()
		{
			auto& modules = this->activeModules();
			for (auto m : modules)
			{
				auto child = TypeInfo::cast<TModule>(m);
				if (child != nullptr)
					return child;
			}
			return nullptr;
		}
		
		// clear pipeline and module
		void clear();

		void pushPersistentModule(std::shared_ptr<Module> m);

		std::list<std::shared_ptr<Module>>& activeModules() {
			if (mModuleUpdated) {
				reconstructPipeline();
			}

			return mModuleList;
		}

		std::map<ObjectId, std::shared_ptr<Module>>& allModules()
		{
			return mModuleMap;
		}

		void enable();
		void disable();

		void updateExecutionQueue();

		void printModuleInfo(bool enabled);

		void forceUpdate();

		/**
		 * Turn a module output field to a node output node
		 */
		void promoteOutputToNode(FBase* base);

		/**
		 * Withdraw a module output field from the node
		 */
		void demoteOutputFromNode(FBase* base);

	protected:
		void preprocess() final;
		void updateImpl() override;

		bool requireUpdate() final;

	private:
		void reconstructPipeline();

	private:
		bool mModuleUpdated = false;
		bool mUpdateEnabled = true;

		std::map<ObjectId, std::shared_ptr<Module>> mModuleMap;
		std::list<std::shared_ptr<Module>>  mModuleList;

		std::list<std::shared_ptr<Module>> mPersistentModule;

		Node* mNode;

		bool mTiming = false;
	};
}

