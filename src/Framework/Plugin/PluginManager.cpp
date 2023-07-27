#include "PluginManager.h"

#include <ghc/fs_std.hpp>

#include <iostream>

namespace dyno
{
	std::shared_ptr<Plugin> Plugin::load(std::string file)
	{
		std::shared_ptr<Plugin> plugin = std::make_shared<Plugin>();
		plugin->mFile = file;

#if !defined(_WIN32)
		plugin->mHnd = ::dlopen(file.c_str(), RTLD_LAZY);
#else
		plugin->mHnd = (void*) ::LoadLibraryA(file.c_str());
#endif 
		if (plugin->mHnd == nullptr) {
			return nullptr;
		}

		plugin->mIsLoaded = true;
#if !defined(_WIN32)
		auto dllEntryPoint =
			reinterpret_cast<PluginEntryFunc>(dlsym(plugin->mHnd, PluginEntryName));
#else
		auto dllEntryPoint =
			reinterpret_cast<PluginEntryFunc>(GetProcAddress((HMODULE)plugin->mHnd, PluginEntryName));
#endif 
		if (dllEntryPoint == nullptr) {
			return nullptr;
		}
		// Retrieve plugin metadata from DLL entry-point function 
		plugin->mEntryPoint = dllEntryPoint();

		return plugin;
	}

	Plugin::Plugin(Plugin&& rhs)
	{
		mIsLoaded = std::move(rhs.mIsLoaded);
		mHnd = std::move(rhs.mHnd);
		mFile = std::move(rhs.mFile);
		mEntryPoint = std::move(rhs.mEntryPoint);
	}

	PluginEntry* Plugin::getInfo() const
	{
		return mEntryPoint;
	}

	bool Plugin::isLoaded() const
	{
		return mIsLoaded;
	}

	void Plugin::unload()
	{
		if (mHnd != nullptr) {
#if !defined(_WIN32)
			::dlclose(mHnd);
#else
			::FreeLibrary((HMODULE)mHnd);
#endif 
			mHnd = nullptr;
			mIsLoaded = false;
		}
	}

	Plugin& Plugin::operator=(Plugin&& rhs)
	{
		std::swap(rhs.mIsLoaded, mIsLoaded);
		std::swap(rhs.mHnd, mHnd);
		std::swap(rhs.mFile, mFile);
		std::swap(rhs.mEntryPoint, mEntryPoint);
		return *this;
	}

	Plugin::~Plugin()
	{
		this->unload();
	}

	std::atomic<PluginManager*> PluginManager::pInstance;
	std::mutex PluginManager::mMutex;

	PluginManager* PluginManager::instance()
	{
		PluginManager* ins = pInstance.load(std::memory_order_acquire);
		if (!ins) {
			std::lock_guard<std::mutex> tLock(mMutex);
			ins = pInstance.load(std::memory_order_relaxed);
			if (!ins) {
				ins = new PluginManager();
				pInstance.store(ins, std::memory_order_release);
			}
		}

		return ins;
	}

	std::string PluginManager::getExtension() const
	{
		std::string ext;
#if defined (_WIN32) 	     
		ext = ".dll"; // Windows 
#elif defined(__unix__) && !defined(__apple__)
		ext = ".so";  // Linux, BDS, Solaris and so on. 
#elif defined(__apple__)
		ext = ".dylib"; // MacOSX 
#else 	 
#error "Not implemented for this platform"
#endif 
		return ext;
	}

	bool PluginManager::loadPlugin(const std::string& pluginName)
	{
		auto plugin = Plugin::load(pluginName);
		if (plugin != nullptr)
		{
			std::cout << "\033[32m\033[1m" << "[Plugin]: loading " << pluginName << " in success " << "\033[0m" << std::endl;
			mPlugins[pluginName] = plugin;

			return true;
		}
		else
		{
			std::cout << "\033[31m\033[1m" << "[Plugin]: loading " << pluginName << " in failure " << "\033[0m" << std::endl;

			return false;
		}
	}

	void PluginManager::loadPluginByPath(const std::string& pathName)
	{
		fs::path file_path(pathName);

		std::error_code error;
		auto file_status = fs::status(file_path, error);
		if (!fs::exists(file_status)) {
			std::cout << "Plugin path do not exist !" << std::endl;
			return;
		}

		for (const auto& entry : fs::directory_iterator(pathName))
		{
			if (entry.path().extension() == getExtension())
			{
				loadPlugin(entry.path().string());
			}
		}
	}

	std::shared_ptr<Plugin> PluginManager::getPlugin(const char* pluginName)
	{
		auto it = mPlugins.find(pluginName);
		if (it == mPlugins.end())
			return nullptr;

		return it->second;
	}
}