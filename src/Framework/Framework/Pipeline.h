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
#include "Framework/Module.h"

namespace dyno
{
	class Node;

	class ModuleIterator
	{
	public:
		ModuleIterator();

		~ModuleIterator();

		ModuleIterator(const ModuleIterator &iterator);

		ModuleIterator& operator= (const ModuleIterator &iterator);

		bool operator== (const ModuleIterator &iterator) const;

		bool operator!= (const ModuleIterator &iterator) const;

		ModuleIterator& operator++ ();
		ModuleIterator& operator++ (int);

		std::shared_ptr<Module> operator *();

		Module* operator->();

		Module* get();

	protected:

		std::weak_ptr<Module> module;

		friend class Pipeline;
	};

	class Pipeline : public Module
	{
	public:
		typedef ModuleIterator Iterator;

		Pipeline(Node* node);
		virtual ~Pipeline();

		Iterator entry();
		Iterator finished();

		unsigned int size();

		void push_back(std::weak_ptr<Module> m);

		void pushModule(std::shared_ptr<Module> m);

		//TODO: this is a temporary method 
		void pushPersistentModule(std::shared_ptr<Module> m);

		std::list<Module*>& activeModules() {
			return mModuleList;
		}

	protected:
		void preprocess() final;
		void updateImpl() override;

		bool requireUpdate() final;

	private:
		void reconstructPipeline();

	private:
		std::weak_ptr<Module> start_module;
		std::weak_ptr<Module> current_module;
		std::weak_ptr<Module> end_module;

		unsigned int num = 0;

		bool mModuleUpdated = false;

		std::map<ObjectId, Module*> mModuleMap;
		std::list<Module*>  mModuleList;

		std::list<Module*> mPersistentModule;

		Node* mNode;
	};
}

