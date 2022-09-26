#include "PluginManager.h"

namespace dyno
{
	Plugin::Plugin(std::string file)
	{
		mFile = std::move(file);
#if !defined(_WIN32)
		mHnd = ::dlopen(mFile.c_str(), RTLD_LAZY);
#else
		mHnd = (void*) ::LoadLibraryA(mFile.c_str());
#endif 
		mIsLoaded = true;
		assert(mHnd != nullptr);
#if !defined(_WIN32)
		auto dllEntryPoint =
			reinterpret_cast<PluginEntryFunc>(dlsym(mHnd, PluginEntryName));
#else
		auto dllEntryPoint =
			reinterpret_cast<PluginEntryFunc>(GetProcAddress((HMODULE)mHnd, PluginEntryName));
#endif 
		assert(dllEntryPoint != nullptr);
		// Retrieve plugin metadata from DLL entry-point function 
		mEntryPoint = dllEntryPoint();
	}

	Plugin::Plugin(Plugin&& rhs)
	{
		mIsLoaded = std::move(rhs.mIsLoaded);
		mHnd = std::move(rhs.mHnd);
		mFile = std::move(rhs.mFile);
		mEntryPoint = std::move(rhs.mEntryPoint);
	}

	IPlugin* Plugin::getInfo() const
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

	IPlugin* PluginManager::loadPlugin(const std::string& pluginName)
	{
		std::string fileName = pluginName + getExtension();
		mPlugins[pluginName] = Plugin(fileName);

		return mPlugins[pluginName].getInfo();
	}

	IPlugin* PluginManager::getPlugin(const char* pluginName)
	{
		auto it = mPlugins.find(pluginName);
		if (it == mPlugins.end())
			return nullptr;

		return it->second.getInfo();
	}
}