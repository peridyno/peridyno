#pragma once
#include "Platform.h"
#include "vulkan/vulkan.h"

#include <vector>
#include <string>
#include <filesystem>

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
#include <android_native_app_glue.h>
#endif

namespace dyno 
{
	class VkContext;

	class VkSystem {

	public:
		static VkSystem* instance();
		static void destroy();

		VkSystem();
		~VkSystem();

		VkContext* currentContext() const;

		bool initialize(bool enableValidation = false);
#if defined(VK_USE_PLATFORM_ANDROID_KHR)
		bool initialize(android_app* state);
#endif
		VkInstance instanceHandle() const;
		VkPhysicalDeviceProperties getDeviceProperties() const;
		VkPhysicalDevice getPhysicalDevice() const;
		std::filesystem::path getAssetPath() const;

		void enableMemoryPool(bool enableMemPool = false);
		void setAssetPath(const std::filesystem::path&);
	private:
		VkResult createVulkanInstance();

		VkContext* ctx;

		bool validation;
		bool useMemoryPool;
		std::string name;
		uint32_t apiVersion;

		VkInstance vkInstance;
		VkPhysicalDevice physicalDevice;
		VkPhysicalDeviceProperties deviceProperties;
		VkPhysicalDeviceFeatures deviceFeatures;
		VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
		void* deviceCreatepNextChain;

		VkDebugUtilsMessengerEXT debugUtilsMessenger;

		std::filesystem::path assertPath;
				
		std::vector<const char*> enabledDeviceExtensions;
		std::vector<const char*> enabledInstanceExtensions;
	};
}