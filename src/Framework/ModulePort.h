/**
 * Copyright 2025 Xiaowei He
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
#include <string>
#include <vector>
#include <memory>

namespace dyno {

	class Module;
	class FCallBackFunc;

	enum ModulePortType
	{
		M_Single,
		M_Multiple,
		M_Unknown
	};

	/*!
	*	\class	ModulePort
	*	\brief	Input ports for Module.
	*/
	class ModulePort
	{
	public:
		ModulePort(std::string name, std::string description, Module* parent = nullptr);
		virtual ~ModulePort() { mModules.clear(); };

		virtual std::string getPortName() { return m_name; };

		ModulePortType getPortType();

		void setPortType(ModulePortType portType);

		virtual std::vector<Module*>& getModules() { return mModules; }


		virtual bool isKindOf(Module* m) = 0;

		virtual bool hasModule(Module* m) = 0;

		inline Module* getParent() { return m_parent; }

		virtual void clear();

		void attach(std::shared_ptr<FCallBackFunc> func);

	protected:
		virtual bool addModule(Module* m) = 0;

		virtual bool removeModule(Module* m) = 0;

		//To call all callback function if connected
		virtual void notify();

		std::vector<Module*> mModules;

	private:

		Module* m_parent = nullptr;

		std::string m_name;
		std::string m_description;
		ModulePortType m_portType;

		std::vector<std::shared_ptr<FCallBackFunc>> mCallbackFunc;

		friend class Module;
	};

	void disconnect(Module* m, ModulePort* port);

	template<typename T>
	class SingleModulePort : public ModulePort
	{
	public:
		SingleModulePort(std::string name, std::string description, Module* parent = nullptr)
			: ModulePort(name, description, parent)
		{
			this->setPortType(ModulePortType::M_Single);
			this->getModules().resize(1);
		};

		~SingleModulePort() override {
			//Disconnect nodes from node ports here instead of inside the destructor of Node to avoid memory leak
			if (mModules[0] != nullptr) {
				disconnect(mModules[0], this);
				//m_nodes[0]->disconnect(this);
			}
			mModules[0] = nullptr; 
		}

		bool isKindOf(Module* m) override
		{
			return nullptr != dynamic_cast<T*>(m) && !hasModule(m);
		}

		bool hasModule(Module* m)
		{
			return mDerivedModule == dynamic_cast<T*>(m);
		}

		std::vector<Module*>& getModules() override
		{
			if (mModules.size() != 1)
				mModules.resize(1);

			mModules[0] = dynamic_cast<Module*>(mDerivedModule);

			return mModules;
		}

		inline T* getDerivedModule()
		{
			return mDerivedModule;
		}

		bool setDerivedModule(T* derived) {
			if (derived != nullptr)
			{
				if (mDerivedModule != nullptr) {
					mDerivedModule = nullptr;
				}

				mDerivedModule = derived;
				mModules[0] = derived;

				return true;
			}
			else
			{
				if (mDerivedModule != nullptr)
				{
					//this->removeNodeFromParent(dynamic_cast<Node*>(m_derived_node));
					mDerivedModule = nullptr;
					mModules[0] = nullptr;
				}
			}

			return false;
		}

	protected:
		bool addModule(Module* m) override
		{
			auto derived = dynamic_cast<T*>(m);
			if (derived != nullptr)
			{
				return setDerivedNode(derived);
			}

			return false;
		}

		bool removeModule(Module* m) override
		{
			mModules[0] = nullptr;
			mDerivedModule = nullptr;

			return true;
		}

	private:
		T* mDerivedModule = nullptr;
	};


	template<typename T>
	class MultipleModulePort : public ModulePort
	{
	public:
		MultipleModulePort(std::string name, std::string description, Module* parent = nullptr)
			: ModulePort(name, description, parent)
		{
			this->setPortType(ModulePortType::M_Multiple);
		};

		~MultipleModulePort() {
			//Disconnect nodes from node ports here instead of inside the destructor of Node to avoid memory leak
			for(auto m : mModules)
			{
				disconnect(m, this);
				//node->disconnect(this);
			}

			mDerivedModules.clear(); 
		}

		void clear() override
		{
			mDerivedModules.clear();

			ModulePort::clear();
		}

		bool addDerivedModule(T* m) {
			if (m != nullptr)
			{
				auto it = find(mDerivedModules.begin(), mDerivedModules.end(), m);

				if (it == mDerivedModules.end())
				{
					mDerivedModules.push_back(m);

					//this->addNodeToParent(dynamic_cast<Node*>(d_node));

					return true;
				}
			}

			return false;
		}

		bool removeDerivedModule(T* m)
		{
			if (m != nullptr)
			{
				auto it = find(mDerivedModules.begin(), mDerivedModules.end(), m);

				if (it != mDerivedModules.end())
				{
					mDerivedModules.erase(it);
					//this->removeNodeFromParent(dynamic_cast<Node*>(d_node));

					return true;
				}
			}

			return false;
		}

		bool isKindOf(Module* m) override
		{
			return nullptr != dynamic_cast<T*>(m) && !hasModule(m);
		}

		bool hasModule(Module* m)
		{
			auto derived = dynamic_cast<T*>(m);
			for (auto m : mDerivedModules)
			{
				if (m == derived)
					return true;
			}

			return false;
		}

		std::vector<Module*>& getModules() override
		{
			mModules.clear();
			mModules.resize(mDerivedModules.size());
			for (int i = 0; i < mModules.size(); i++)
			{
				mModules[i] = dynamic_cast<Module*>(mDerivedModules[i]);
			}
			return mModules;
		}

		inline std::vector<T*>& getDerivedModules()
		{
			return mDerivedModules;
		}

	protected:
		bool addModule(Module* m) override {
			auto derived = dynamic_cast<T*>(m);
			if (derived != nullptr)
			{
				addDerivedModule(derived);
			}

			return false;
		}

		bool removeModule(Module* m)  override
		{
			auto derived = dynamic_cast<T*>(m);

			if (derived != nullptr)
			{
				return removeDerivedModule(derived);
			}

			return false;
		}

	private:
		std::vector<T*> mDerivedModules;
	};
}
