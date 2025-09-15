#include "ModulePort.h"
#include "Module.h"

namespace dyno
{

	ModulePort::ModulePort(std::string name, std::string description, Module* parent /*= nullptr*/)
		: m_name(name)
		, m_description(description)
		, m_portType(ModulePortType::M_Unknown)
		, m_parent(parent)
	{
		parent->addModulePort(this);
	}

	ModulePortType ModulePort::getPortType()
	{
		return m_portType;
	}

	void ModulePort::setPortType(ModulePortType portType)
	{
		m_portType = portType;
	}

	void ModulePort::clear()
	{
		mModules.clear();
	}

	void ModulePort::attach(std::shared_ptr<FCallBackFunc> func)
	{
		mCallbackFunc.push_back(func);
	}

	void ModulePort::notify()
	{
		for (auto func : mCallbackFunc)
		{
			func->update();
		}
	}

	void disconnect(Module* m, ModulePort* port)
	{
		m->disconnect(port);
	}

}

