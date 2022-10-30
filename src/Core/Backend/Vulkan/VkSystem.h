#pragma once
#include "Platform.h"
#include "vulkan/vulkan.h"

#include <vector>
#include <string>

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
#include <android_native_app_glue.h>
#endif

namespace px {
	class VkContext;

	class VkSystem {

	public:
		static VkSystem* instance();

		VkContext* currentContext() { return ctx; }

		bool initialize(bool enableValidation = false);
#if defined(VK_USE_PLATFORM_ANDROID_KHR)
		bool initialize(android_app* state);
#endif
		VkInstance instanceHandle() { return vkInstance; }

		void enableMemoryPool(bool enableMemPool = false) {
			useMemoryPool = enableMemPool;
		}

		VkPhysicalDeviceProperties getDeviceProperties() {
			return deviceProperties;
		}

		VkPhysicalDevice getPhysicalDevice() {
			return physicalDevice;
		}

	private:
		VkSystem();
		~VkSystem();

		/*!
		 *	\brief	Creates the application wide Vulkan instance.
		 */
		VkResult createVulkanInstance();

		/*!
		 *	\brief	Current Vulkan context.
		 */
		VkContext* ctx = nullptr;

		bool validation;
		bool useMemoryPool = true;
		std::string name = "Vulkan";
		uint32_t apiVersion = VK_API_VERSION_1_1;

		/*!
		 *	\brief	Vulkan instance, stores all per-application states.
		 */
		VkInstance vkInstance;
		VkPhysicalDevice physicalDevice;
		VkPhysicalDeviceProperties deviceProperties;
		VkPhysicalDeviceFeatures deviceFeatures;
		VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
		VkPhysicalDeviceFeatures enabledFeatures{};
		void* deviceCreatepNextChain = nullptr;

		std::vector<const char*> enabledDeviceExtensions;

		std::vector<const char*> enabledInstanceExtensions;

		std::vector<std::string> supportedInstanceExtensions;
	};
}