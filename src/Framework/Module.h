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
#include "Platform.h"
#include <memory>
#include <vector>
#include <cassert>
#include <iostream>
#include "OBase.h"
#include "Log.h"
#include "DataTypes.h"
#include "DeclareEnum.h"
#include "DeclareField.h"
#include "FCallbackFunc.h"
#include "FieldTypes.h"

namespace dyno
{
	class SceneGraph;
	class Node;

	class Module : public OBase
	{
	public:
		Module(std::string name = "default");

		~Module(void) override;

		bool initialize();

		void update();

		void setName(std::string name);

		std::string getName() override;

		/**
		 * @brief Set the parent node
		 */
		virtual void setParentNode(Node* node);

		/**
		 * @breif Return the Node that contains this Module
		 * 
		 * @return return nullptr if the Module is not contained in any Node
		 */
		Node* getParentNode();

		/**
		 * @breif Return the SceneGraph that contains this Module
		 *
		 * @return return nullptr if the module is not contained in any node
		 */
		SceneGraph* getSceneGraph();

		bool isInitialized();

		virtual std::string getModuleType() { return "Module"; }

		bool attachField(FBase* field, std::string name, std::string desc, bool autoDestroy = true) override;

		/**
		 * @brief Check the completeness of input fields
		 *
		 * @return true, if all input fields are appropriately set.
		 * @return false, if any of the input field is empty.
		 */
		bool isInputComplete();
		bool isOutputCompete();


	public:
		DEF_VAR(bool, ForceUpdate, false, "");

		/**
		 * @brief Set the update strategy for the module
		 *
		 * @param b true, the module will be updated no matter the control and input variables are changed;
		 * 			false, the module will updated only when one of the variables are changed.
		 */
		void setUpdateAlways(bool b);

	protected:
		//TODO: remove this step
		virtual bool initializeImpl();
		virtual void updateImpl();


		virtual void preprocess() {};

		virtual void postprocess() {};

		virtual bool validateInputs();
		virtual bool validateOutputs();

		virtual bool requireUpdate();

		/**
		 * @brief Two functions called at the beginning and end of update()
		 *	used for debug
		 */
		virtual void updateStarted();
		virtual void updateEnded();

	private:
		Node* m_node;
		std::string m_module_name;
		bool m_initialized;
	};
}