#pragma once
#include <string> 
#include <map>
#include <vector>

/** Macro makes a symbol visible. */
#if defined(_WIN32)
  // MS-Windows NT 
  #define DYNO_PLUGIN_EXPORT extern "C" __declspec(dllexport) 
#else
  // Unix-like OSes
  #define DYNO_PLUGIN_EXPORT extern "C" __attribute__ ((visibility ("default")))
#endif 

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

		void initialize();

		virtual void initializeNodeCreators() {};

	public:
		/** Plugin name */
		std::string m_name;

		/** Plugin version */
		std::string m_version;
	};
}
