#include "Module.h"
#include "Node.h"
#include "SceneGraph.h"

//#define PYTHON

namespace dyno
{
	Module::Module(std::string name)
		: OBase()
		, m_node(nullptr)
		, m_initialized(false)
	{
		m_module_name = name;
	}

	Module::~Module(void)
	{
		mImportModules.clear();
		mExportModules.clear();
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
#ifdef PYTHON
		std::cout << "Module::update" << std::endl;
#endif // PYTHON

		if (!isInitialized())
		{
			bool ret = initialize();
			if (ret == false)
				return;
		}

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

			//reset parameters
			for (auto param : fields_param)
			{
				param->tack();
			}

			//reset input fields
			for (auto f_in : fields_input)
			{
				f_in->tack();
			}

			//tag all output fields as modifed
			for (auto f_out : fields_output)
			{
				f_out->tick();
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
		for (auto f_in : fields_input)
		{
			if (!f_in->isOptional() && f_in->isEmpty())
			{
				Node* par = this->getParentNode();
				if (par != nullptr && par->getSceneGraph() != nullptr && par->getSceneGraph()->isValidationInfoPrintable())
				{
					std::string errMsg = std::string("The input field ") + f_in->getObjectName() +
						std::string(" in Module ") + this->getClassInfo()->getClassName() + std::string(" is not set!");

					Log::sendMessage(Log::Error, errMsg);
				}

				return false;
			}
		}

		return true;
	}

	bool Module::isOutputCompete()
	{
		//If any output field is empty, return false;
		for (auto f_out : fields_output)
		{
			if (f_out->isEmpty())
			{
				Node* par = this->getParentNode();
				if (par != nullptr && par->getSceneGraph() != nullptr && par->getSceneGraph()->isValidationInfoPrintable())
				{
					std::string errMsg = std::string("The output field ") + f_out->getObjectName() +
						std::string(" in Module ") + this->getClassInfo()->getClassName() + std::string(" is not prepared!");

					Log::sendMessage(Log::Error, errMsg);
				}
				return false;
			}
		}

		return true;
	}

	bool Module::connect(ModulePort* nPort)
	{
		if (!this->allowExported() || !nPort->getParent()->allowImported())
			return false;

		nPort->notify();

		return this->appendExportModule(nPort);
	}

	bool Module::disconnect(ModulePort* nPort)
	{
		return this->removeExportModule(nPort);
	}

	void Module::updateStarted()
	{
	}

	void Module::updateEnded()
	{
	}

	bool Module::appendExportModule(ModulePort* nodePort)
	{
		auto it = find(mExportModules.begin(), mExportModules.end(), nodePort);
		if (it != mExportModules.end()) {
			//Log::sendMessage(Log::Info, FormatConnectionInfo(this, nodePort, true, false));
			return false;
		}

		mExportModules.push_back(nodePort);

		//Log::sendMessage(Log::Info, FormatConnectionInfo(this, nodePort, true, true));
		return nodePort->addModule(this);
	}

	bool Module::removeExportModule(ModulePort* nodePort)
	{
		//TODO: this is a hack, otherwise the app will crash
		if (mExportModules.size() == 0) {
			return false;
		}

		auto it = find(mExportModules.begin(), mExportModules.end(), nodePort);
		if (it == mExportModules.end()) {
			//Log::sendMessage(Log::Info, FormatConnectionInfo(this, nodePort, false, false));
			return false;
		}

		mExportModules.erase(it);

		//Log::sendMessage(Log::Info, FormatConnectionInfo(this, nodePort, false, true));
		return nodePort->removeModule(this);
	}

	bool Module::addModulePort(ModulePort* port)
	{
		mImportModules.push_back(port);

		return true;
	}

	bool Module::validateOutputs()
	{
		return isOutputCompete();
	}

	bool Module::requireUpdate()
	{
		if (this->varForceUpdate()->getValue())
		{
			return true;
		}

		//check input fields
		bool modified = false;
		for (auto f_in : fields_input)
		{
			modified |= f_in->isModified();
		}

		//check control fields
		for (auto var : fields_param)
		{
			modified |= var->isModified();
		}

		return modified;
	}

	void Module::setName(std::string name)
	{
		//m_module_name.setValue(name);
		m_module_name = name;
	}

	void Module::setParentNode(Node* node)
	{
		m_node = node;
	}

	std::string Module::getName()
	{
		return m_module_name;
	}

	Node* Module::getParentNode()
	{
		if (m_node == NULL) {
			Log::sendMessage(Log::Error, "Parent node is not set!");
		}

		return m_node;
	}

	dyno::SceneGraph* Module::getSceneGraph()
	{
		auto node = this->getParentNode();

		if (node == NULL) return NULL;

		return node->getSceneGraph();
	}

	void Module::setUpdateAlways(bool b)
	{
		this->varForceUpdate()->setValue(b);
	}

	bool Module::isInitialized()
	{
		return m_initialized;
	}

	bool Module::initializeImpl()
	{
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

		case FieldTypeEnum::IO:
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