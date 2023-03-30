#pragma once
#include <string> 
#include <map>
#include <vector>

#include <atomic>
#include <mutex>

#include <Platform.h>

namespace dyno
{
	struct PluginEntry
	{
	public:
		PluginEntry();

		/** Get Plugin Name */
		const char* name() const;

		/** Get Plugin Version */
		const char* version() const;

		/** Get Plugin Description */
		const char* description() const;

		void setName(std::string pluginName);

		void setVersion(std::string pluginVersion);

		void setDescription(std::string desc);

		bool initialize();

	protected:
		virtual void initializeNodeCreators() {};

		virtual void initializeActions() {};

	private:
		/** Plugin name */
		std::string mName;

		/** Plugin version */
		std::string mVersion;

		/** Plugin description */
		std::string mDescription;

		bool mInitialized = false;
	};
}
