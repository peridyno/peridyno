#include "Module.h"
#include "Node.h"

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
// 		//Before deallocating data, fields should be disconnected first
// 		for each (auto f in fields_input)
// 		{
// 			FBase* src = f->getSource();
// 			if (src != nullptr) {
// 				src->disconnectField(f);
// 			}
// 		}
// 
// 		for each (auto f in fields_output)
// 		{
// 			auto& sinks = f->getSinks();
// 			for each (auto sink in sinks)
// 			{
// 				f->disconnectField(sink);
// 			}
// 		}
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
		this->updateStarted();

		if (!this->validateInputs()) {
			return;
		}

		if (this->requireUpdate()) {
			//pre processing
			this->preprocess();

			//do execution if any field is modified
			this->updateImpl();

			//post processing
			this->postprocess();

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

		if (!this->validateOutputs()) {
			return;
		}

		this->updateEnded();
	}

	bool Module::validateInputs()
	{
		return isInputComplete();
	}

	bool Module::isInputComplete()
	{
		//If any input field is empty, return false;
		for each (auto f_in in fields_input)
		{
			if (!f_in->isOptional() && f_in->isEmpty())
			{
				std::string errMsg = std::string("The field ") + f_in->getObjectName() +
					std::string(" in Module ") + this->getClassInfo()->getClassName() + std::string(" is not set!");

				std::cout << errMsg << std::endl;
				return false;
			}
		}

		return true;
	}

	bool Module::isOutputCompete()
	{
		//If any output field is empty, return false;
		for each (auto f_out in fields_output)
		{
			if (f_out->isEmpty())
			{
				std::string errMsg = std::string("The field ") + f_out->getObjectName() +
					std::string(" in Module ") + this->getClassInfo()->getClassName() + std::string(" is not prepared!");

				Log::sendMessage(Log::Error, errMsg);
				return false;
			}
		}

		return true;
	}

	void Module::updateStarted()
	{

	}

	void Module::updateEnded()
	{

	}

	bool Module::validateOutputs()
	{
		return isOutputCompete();
	}

	//TODO: check whether any of the input fields is updated
	bool Module::requireUpdate()
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

	bool Module::initializeImpl()
	{
		if (m_node == nullptr)
		{
			Log::sendMessage(Log::Warning, "Parent is not set");
			return false;
		}

		return true;
	}

	void Module::updateImpl()
	{
	}

	bool Module::attachField(FBase* field, std::string name, std::string desc, bool autoDestroy)
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