#include "PluginEntry.h"

namespace dyno
{
	PluginEntry::PluginEntry() :
		mName("Default"),
		mVersion("1.0")
	{
	}

	const char* PluginEntry::name() const
	{
		return mName.data();
	}

	const char* PluginEntry::version() const
	{
		return mVersion.data();
	}

	const char* PluginEntry::description() const
	{
		return mDescription.data();
	}

	void PluginEntry::setName(std::string pluginName)
	{
		mName = pluginName;
	}

	void PluginEntry::setVersion(std::string pluginVersion)
	{
		mVersion = pluginVersion;
	}

	void PluginEntry::setDescription(std::string desc)
	{
		mDescription = desc;
	}

	bool PluginEntry::initialize()
	{
		if (!mInitialized)
		{
			this->initializeNodeCreators();
			this->initializeActions();
			mInitialized = true;

			return true;
		}

		return false;
	}
}