#pragma once
#include <string> 
#include <map>
#include <vector>

#include <atomic>
#include <mutex>

#include <Platform.h>

namespace dyno
{
	struct IPlugin
	{
	public:
		IPlugin();

		/** Get Plugin Name */
		const char* name() const;

		/** Get Plugin Version */
		const char* version() const;

		bool initialize();

		virtual void initializeNodeCreators() {};

	public:
		/** Plugin name */
		std::string m_name;

		/** Plugin version */
		std::string m_version;

		bool mInitialized = false;
	};
}
