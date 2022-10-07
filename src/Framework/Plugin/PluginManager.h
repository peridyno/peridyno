#pragma once
#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <atomic>
#include <mutex>

#include "PluginEntry.h"

// Unix
#if defined(_WIN32)
   #include <windows.h>
#elif defined(__unix__)
  // APIs: dlopen, dlclose, dlopen 
  #include <dlfcn.h>
#else
  #error "Not supported operating system"
#endif 

namespace dyno 
{
	/** @brief Class form managing and encapsulating shared libraries loading  */
	class Plugin
	{
	public:
		Plugin() {}

		~Plugin();

		Plugin(Plugin const&) = delete;
		Plugin& operator=(const Plugin&) = delete;

		Plugin(Plugin&& rhs);

		Plugin& operator=(Plugin&& rhs);

		PluginEntry* getInfo() const;

		bool isLoaded() const;

		void unload();

		static std::shared_ptr<Plugin> load(std::string file);

	private:
		/** @brief Function pointer to DLL entry-point */
		using PluginEntryFunc = PluginEntry * (*) ();

		/** @brief Name of DLL entry point that a Plugin should export */
		static constexpr const char* PluginEntryName = "initDynoPlugin";

		/** @brief Shared library handle */
		void* mHnd = nullptr;

		/** @brief Shared library file name */
		std::string  mFile = "";

		/** @brief Flag to indicate whether plugin (shared library) is loaded into current process. */
		bool         mIsLoaded = false;

		/** @brief Pointer to shared library factory class returned by the DLL entry-point function */
		PluginEntry* mEntryPoint = nullptr;
	};

	/** 
	 * @brief Repository of plugins.
	 * It can instantiate any class from any loaded plugin by its name.
	 **/
	class PluginManager
	{
	public:
		static PluginManager* instance();

		std::string getExtension() const;

		bool loadPlugin(const std::string& pluginName);

		void loadPluginByPath(const std::string& pathName);

		std::shared_ptr<Plugin> getPlugin(const char* pluginName);

	private:
		PluginManager() {};

		using PluginMap = std::map<std::string, std::shared_ptr<Plugin>>;

		static std::atomic<PluginManager*> pInstance;
		static std::mutex mMutex;

		PluginMap mPlugins;
	};
}
