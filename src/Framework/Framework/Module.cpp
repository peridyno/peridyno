#include "Module.h"
#include "Framework/Node.h"

namespace dyno
{

Module::Module(std::string name)
	: m_node(nullptr)
	, m_initialized(false)
{
//	attachField(&m_module_name, "module_name", "Module name", false);

//	m_module_name.setValue(name);
	m_module_name = name;
}

Module::~Module(void)
{

}

bool Module::initialize()
{
	if (m_initialized)
	{
		return true;
	}
	m_initialized = initializeImpl();

	return m_initialized;
}

void Module::update()
{
	if (!isInputComplete())
	{
		Log::sendMessage(Log::Error, std::string("Input for ") + this->getName() + std::string(" with class name of ") + this->getClassInfo()->getClassName() + std::string(" should be appropriately set"));
		return;
	}

	if (m_update_required)
	{
		//pre processing
		this->begin();

		//do execution if any field is modified
		this->execute();

		//post processing
		this->end();

		//reset input fields
		for each (auto f_in in fields_input)
		{
			f_in->tagModified(false);
		}

		//tag all output fields as modifed
		for each (auto f_out in fields_output)
		{
			f_out->tagModified(true);
		}
	}
}

bool Module::isInputComplete()
{
	//If any input field is empty, return false;
	for each (auto f_in in fields_input)
	{
		if (!f_in->isOptional() && f_in->isEmpty())
		{
			Log::sendMessage(Log::Error, std::string("The field ") + f_in->getObjectName() + std::string(" in Module ") + this->getClassInfo()->getClassName() + std::string(" is not set!"));
			return false;
		}
	}

	return true;
}

bool Module::execute()
{
	return true;
}

void Module::setName(std::string name)
{
	//m_module_name.setValue(name);
	m_module_name = name;
}

void Module::setParent(Node* node)
{
	m_node = node;
}

std::string Module::getName()
{
	return m_module_name;
}

bool Module::isInitialized()
{
	return m_initialized;
}

bool Module::findInputField(FieldBase* field)
{
	auto result = find(fields_input.begin(), fields_input.end(), field);
	// return false if no field is found!
	if (result == fields_input.end())
	{
		return false;
	}
	return true;
}

bool Module::addInputField(FieldBase* field)
{
	if (findInputField(field))
	{
		return false;
	}

	this->addField(field);

	fields_input.push_back(field);

	return true;
}

bool Module::removeInputField(FieldBase* field)
{
	if (!findInputField(field))
	{
		return false;
	}

	this->removeField(field);

	auto result = find(fields_input.begin(), fields_input.end(), field);
	if (result != fields_input.end())
	{
		fields_input.erase(result);
	}

	return true;
}

bool Module::findOutputField(FieldBase* field)
{
	auto result = find(fields_output.begin(), fields_output.end(), field);
	// return false if no field is found!
	if (result == fields_output.end())
	{
		return false;
	}
	return true;
}

bool Module::addOutputField(FieldBase* field)
{
	if (findOutputField(field))
	{
		return false;
	}

	this->addField(field);

	fields_output.push_back(field);

	return true;
}

bool Module::removeOutputField(FieldBase* field)
{
	if (!findOutputField(field))
	{
		return false;
	}

	this->removeField(field);

	auto result = find(fields_output.begin(), fields_output.end(), field);
	if (result != fields_output.end())
	{
		fields_output.erase(result);
	}

	return true;
}

bool Module::findParameter(FieldBase* field)
{
	auto result = find(fields_param.begin(), fields_param.end(), field);
	// return false if no field is found!
	if (result == fields_param.end())
	{
		return false;
	}
	return true;
}

bool Module::addParameter(FieldBase* field)
{
	if (findParameter(field))
	{
		return false;
	}

	this->addField(field);

	fields_param.push_back(field);

	return true;
}

bool Module::removeParameter(FieldBase* field)
{
	if (!findParameter(field))
	{
		return false;
	}

	this->removeField(field);

	auto result = find(fields_param.begin(), fields_param.end(), field);
	if (result != fields_output.end())
	{
		fields_param.erase(result);
	}

	return true;
}

bool Module::initializeImpl()
{
	if (m_node == nullptr)
	{
		Log::sendMessage(Log::Warning, "Parent is not set");
		return false;
	}

	return true;
}

bool Module::attachField(FieldBase* field, std::string name, std::string desc, bool autoDestroy)
{
	field->setParent(this);
	field->setObjectName(name);
	field->setDescription(desc);
	field->setAutoDestroy(autoDestroy);


	bool ret = false;
	auto fType = field->getFieldType();
	switch (field->getFieldType())
	{
	case FieldTypeEnum::In:
		ret = addInputField(field);
		break;

	case FieldTypeEnum::Out:
		ret = addOutputField(field);
		break;

	case FieldTypeEnum::Param:
		ret = addParameter(field);
		break;

	default:
		break;
	}

	if (!ret)
	{
		Log::sendMessage(Log::Error, std::string("The field ") + name + std::string(" already exists!"));
	}
	return ret;
}

}