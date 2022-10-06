#include "PluginInterface.h"

namespace dyno
{
	IPlugin::IPlugin() :
		m_name("Default"),
		m_version("1.0")
	{
	}

	const char* IPlugin::name() const
	{
		return m_name.data();
	}

	const char* IPlugin::version() const
	{
		return m_version.data();
	}

	bool IPlugin::initialize()
	{
		if (!mInitialized)
		{
			this->initializeNodeCreators();
			mInitialized = true;

			return true;
		}

		return false;
	}
}