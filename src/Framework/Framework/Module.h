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
#include "Base.h"
#include "Log.h"
#include "DataTypes.h"
#include "DeclareEnum.h"
#include "DeclareField.h"
#include "../FieldTypes.h"

namespace dyno
{
class Node;

class Module : public Base
{
public:
	Module(std::string name = "default");

	~Module(void) override;

	bool initialize();

	void update();

	void setName(std::string name);
	void setParent(Node* node);

	std::string getName();

	Node* getParent() {
		if (m_node == NULL) {
			Log::sendMessage(Log::Error, "Parent node is not set!");
		}

		return m_node;
	}

	bool isInitialized();

	virtual std::string getModuleType() { return "Module"; }

	bool findInputField(FBase* field);
	bool addInputField(FBase* field);
	bool removeInputField(FBase* field);

	std::vector<FBase*>& getInputFields() { return fields_input; }

	bool findOutputField(FBase* field);
	bool addOutputField(FBase* field);
	bool removeOutputField(FBase* field);

	std::vector<FBase*>& getOutputFields() { return fields_output; }

	bool findParameter(FBase* field);
	bool addParameter(FBase* field);
	bool removeParameter(FBase* field);

	std::vector<FBase*>& getParameters() { return fields_param; }

	virtual std::weak_ptr<Module> next() { return m_module_next; }

	void setNext(std::weak_ptr<Module> next_module) { m_module_next = next_module; }

	bool attachField(FBase* field, std::string name, std::string desc, bool autoDestroy = true) override;
protected:
	/// \brief Initialization function for each module
	/// 
	/// This function is used to initialize internal variables for each module
	/// , it is called after all fields are set.
	virtual bool initializeImpl();
	virtual void updateImpl();
	

	virtual void preprocess() {};

	virtual void postprocess() {};

	virtual bool validateInputs();
	virtual bool validateOutputs();

	virtual bool requireUpdate();

	/**
	 * @brief Check the completeness of input fields
	 *
	 * @return true, if all input fields are appropriately set.
	 * @return false, if any of the input field is empty.
	 */
	bool isInputComplete();
	bool isOutputCompete();

private:
	Node* m_node;
	std::string m_module_name;
	bool m_initialized;

	bool m_update_required = true;

	std::weak_ptr<Module> m_module_next;

	std::vector<FBase*> fields_input;
	std::vector<FBase*> fields_output;
	std::vector<FBase*> fields_param;
};
}