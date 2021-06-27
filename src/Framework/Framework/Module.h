#pragma once
#include "Platform.h"
#include <memory>
#include <vector>
#include <cassert>
#include <iostream>
#include "Base.h"
#include "Log.h"
#include "DataTypes.h"
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

	virtual bool execute();

	/**
	 * @brief Check the completeness of input fields
	 * 
	 * @return true, if all input fields are appropriately set.
	 * @return false, if any of the input field is empty.
	 */
	bool isInputComplete();

	void setName(std::string name);
	void setParent(Node* node);

	std::string getName();

	Node* getParent()
	{
		if (m_node == NULL)
		{
			Log::sendMessage(Log::Error, "Parent node is not set!");
		}
		return m_node;
	}

	bool isInitialized();

	virtual std::string getModuleType() { return "Module"; }

	bool findInputField(FieldBase* field);
	bool addInputField(FieldBase* field);
	bool removeInputField(FieldBase* field);

	std::vector<FieldBase*>& getInputFields() { return fields_input; }

	bool findOutputField(FieldBase* field);
	bool addOutputField(FieldBase* field);
	bool removeOutputField(FieldBase* field);

	std::vector<FieldBase*>& getOutputFields() { return fields_output; }

	bool findParameter(FieldBase* field);
	bool addParameter(FieldBase* field);
	bool removeParameter(FieldBase* field);

	std::vector<FieldBase*>& getParameters() { return fields_param; }

	virtual std::weak_ptr<Module> next() { return m_module_next; }

	void setNext(std::weak_ptr<Module> next_module) { m_module_next = next_module; }

	bool attachField(FieldBase* field, std::string name, std::string desc, bool autoDestroy = true) override;


protected:
	/// \brief Initialization function for each module
	/// 
	/// This function is used to initialize internal variables for each module
	/// , it is called after all fields are set.
	virtual bool initializeImpl();

	virtual void begin() {};

	virtual void end() {};

	std::weak_ptr<Module> m_module_next;

private:
	Node* m_node;
	std::string m_module_name;
	bool m_initialized;

	bool m_update_required = true;

	std::vector<FieldBase*> fields_input;
	std::vector<FieldBase*> fields_output;
	std::vector<FieldBase*> fields_param;
};
}