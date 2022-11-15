#include "VkSystem.h"
#include "VulkanTools.h"
#include "VulkanDebug.h"
#include "VkContext.h"

#include <iostream>

namespace px {

	VkSystem* VkSystem::instance()
	{
		static VkSystem gInstance;
		return &gInstance;
	}

	VkSystem::VkSystem()
	{
	}

	VkSystem::~VkSystem()
	{
		if (validation)
		{
			vks::debug::freeDebugCallback(vkInstance);
		}

		if (ctx != nullptr) {
			delete ctx;
		}

		vkDestroyInstance(vkInstance, nullptr);
	}

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
	bool VkSystem::initialize(android_app* state) {
		androidApp = state;
		return VkSystem::initialize();
	}
#endif

	bool VkSystem::initialize(bool enableValidation)
	{
		validation = enableValidation;

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
		// Vulkan library is loaded dynamically on Android
	bool libLoaded = vks::android::loadVulkanLibrary();
	assert(libLoaded);
#endif

		VkResult err;

		// Vulkan instance
		err = createVulkanInstance();
		if (err) {
			vks::tools::exitFatal("Could not create Vulkan instance : \n" + vks::tools::errorString(err), err);
			return false;
		}

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
		vks::android::loadVulkanFunctions(vkInstance);
#endif

		// If requested, we enable the default validation layers for debugging
		if (validation)
		{
			// The report flags determine what type of messages for the layers will be displayed
			// For validating (debugging) an application the error and warning bits should suffice
			VkDebugReportFlagsEXT debugReportFlags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
			// Additional flags include performance info, loader and layer debug messages, etc.
			vks::debug::setupDebugging(vkInstance, debugReportFlags, VK_NULL_HANDLE);
		}

		// Physical device
		uint32_t gpuCount = 0;
		// Get number of available physical devices
		VK_CHECK_RESULT(vkEnumeratePhysicalDevices(vkInstance, &gpuCount, nullptr));
		assert(gpuCount > 0);
		// Enumerate devices
		std::vector<VkPhysicalDevice> physicalDevices(gpuCount);
		err = vkEnumeratePhysicalDevices(vkInstance, &gpuCount, physicalDevices.data());
		if (err) {
			vks::tools::exitFatal("Could not enumerate physical devices : \n" + vks::tools::errorString(err), err);
			return false;
		}

		// GPU selection

		// Select physical device to be used for the Vulkan example
		// Defaults to the first device unless specified by command line
		uint32_t selectedDevice = 0;
/*
#if !defined(VK_USE_PLATFORM_ANDROID_KHR)
		// GPU selection via command line argument
		for (size_t i = 0; i < args.size(); i++)
		{
			// Select GPU
			if ((args[i] == std::string("-g")) || (args[i] == std::string("-gpu")))
			{
				char* endptr;
				uint32_t index = strtol(args[i + 1], &endptr, 10);
				if (endptr != args[i + 1])
				{
					if (index > gpuCount - 1)
					{
						std::cerr << "Selected device index " << index << " is out of range, reverting to device 0 (use -listgpus to show available Vulkan devices)" << "\n";
					}
					else
					{
						std::cout << "Selected Vulkan device " << index << "\n";
						selectedDevice = index;
					}
				};
				break;
			}
			// List available GPUs
			if (args[i] == std::string("-listgpus"))
			{
				uint32_t gpuCount = 0;
				VK_CHECK_RESULT(vkEnumeratePhysicalDevices(vkInstance, &gpuCount, nullptr));
				if (gpuCount == 0)
				{
					std::cerr << "No Vulkan devices found!" << "\n";
				}
				else
				{
					// Enumerate devices
					std::cout << "Available Vulkan devices" << "\n";
					std::vector<VkPhysicalDevice> devices(gpuCount);
					VK_CHECK_RESULT(vkEnumeratePhysicalDevices(vkInstance, &gpuCount, devices.data()));
					for (uint32_t j = 0; j < gpuCount; j++) {
						VkPhysicalDeviceProperties deviceProperties;
						vkGetPhysicalDeviceProperties(devices[j], &deviceProperties);
						std::cout << "Device [" << j << "] : " << deviceProperties.deviceName << std::endl;
						std::cout << " Type: " << vks::tools::physicalDeviceTypeString(deviceProperties.deviceType) << "\n";
						std::cout << " API: " << (deviceProperties.apiVersion >> 22) << "." << ((deviceProperties.apiVersion >> 12) & 0x3ff) << "." << (deviceProperties.apiVersion & 0xfff) << "\n";
					}
				}
			}
		}
#endif*/

		physicalDevice = physicalDevices[selectedDevice];

		// Store properties (including limits), features and memory properties of the physical device (so that examples can check against them)
		vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
		vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &deviceMemoryProperties);

		// Derived examples can override this to set actual features (based on above readings) to enable for logical device creation
		//getEnabledFeatures();

		// Vulkan device creation
		// This is handled by a separate class that gets a logical device representation
		// and encapsulates functions related to a device
		ctx = new VkContext(physicalDevice);
		VkResult res = ctx->createLogicalDevice(enabledFeatures, enabledDeviceExtensions, deviceCreatepNextChain);
		if (res != VK_SUCCESS) {
			vks::tools::exitFatal("Could not create Vulkan device: \n" + vks::tools::errorString(res), res);
			return false;
		}

		if (useMemoryPool) {
			res = ctx->createMemoryPool(vkInstance, apiVersion);
			if (res != VK_SUCCESS) {
				vks::tools::exitFatal("Could not create Vulkan memory pool: \n" + vks::tools::errorString(res), res);
				return false;
			}
		}

		return true;
	}

	VkResult VkSystem::createVulkanInstance()
	{
		// Validation can also be forced via a define
#if defined(_VALIDATION)
		this->validation = true;
#endif

		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = name.c_str();
		appInfo.pEngineName = name.c_str();
		appInfo.apiVersion = apiVersion;

		std::vector<const char*> instanceExtensions = { VK_KHR_SURFACE_EXTENSION_NAME };

		// Enable surface extensions depending on os
#if defined(_WIN32)
		instanceExtensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
		instanceExtensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#elif defined(_DIRECT2DISPLAY)
		instanceExtensions.push_back(VK_KHR_DISPLAY_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_DIRECTFB_EXT)
		instanceExtensions.push_back(VK_EXT_DIRECTFB_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
		instanceExtensions.push_back(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_XCB_KHR)
		instanceExtensions.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_IOS_MVK)
		instanceExtensions.push_back(VK_MVK_IOS_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_MACOS_MVK)
		instanceExtensions.push_back(VK_MVK_MACOS_SURFACE_EXTENSION_NAME);
#endif

		// Get extensions supported by the instance and store for later use
		uint32_t extCount = 0;
		std::vector<std::string> supportedInstanceExtensions;

		vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
		if (extCount > 0)
		{
			std::vector<VkExtensionProperties> extensions(extCount);
			if (vkEnumerateInstanceExtensionProperties(nullptr, &extCount, &extensions.front()) == VK_SUCCESS)
			{
				for (VkExtensionProperties extension : extensions)
				{
					supportedInstanceExtensions.push_back(extension.extensionName);
				}
			}
		}

		// Enabled requested instance extensions
		if (enabledInstanceExtensions.size() > 0)
		{
			for (const char * enabledExtension : enabledInstanceExtensions)
			{
				// Output message if requested extension is not available
				if (std::find(supportedInstanceExtensions.begin(), supportedInstanceExtensions.end(), enabledExtension) == supportedInstanceExtensions.end())
				{
					std::cerr << "Enabled instance extension \"" << enabledExtension << "\" is not present at instance level\n";
				}
				instanceExtensions.push_back(enabledExtension);
			}
		}

		VkInstanceCreateInfo instanceCreateInfo = {};
		instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		instanceCreateInfo.pNext = NULL;
		instanceCreateInfo.pApplicationInfo = &appInfo;
		if (instanceExtensions.size() > 0)
		{
			if (validation)
			{
				instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
			}
			instanceCreateInfo.enabledExtensionCount = (uint32_t)instanceExtensions.size();
			instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions.data();
		}
		if (validation)
		{
			// The VK_LAYER_KHRONOS_validation contains all current validation functionality.
			// Note that on Android this layer requires at least NDK r20
			const char* validationLayerName = "VK_LAYER_KHRONOS_validation";
			// Check if this layer is available at instance level
			uint32_t instanceLayerCount;
			vkEnumerateInstanceLayerProperties(&instanceLayerCount, nullptr);
			std::vector<VkLayerProperties> instanceLayerProperties(instanceLayerCount);
			vkEnumerateInstanceLayerProperties(&instanceLayerCount, instanceLayerProperties.data());
			bool validationLayerPresent = false;
			for (VkLayerProperties layer : instanceLayerProperties) {
				if (strcmp(layer.layerName, validationLayerName) == 0) {
					validationLayerPresent = true;
					break;
				}
			}
			if (validationLayerPresent) {
				instanceCreateInfo.ppEnabledLayerNames = &validationLayerName;
				instanceCreateInfo.enabledLayerCount = 1;
			}
			else {
				std::cerr << "Validation layer VK_LAYER_KHRONOS_validation not present, validation is disabled";
			}

			VkValidationFeatureEnableEXT enabled[] = { VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT };
			VkValidationFeaturesEXT features{ VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT };
			features.disabledValidationFeatureCount = 0;
			features.enabledValidationFeatureCount = 1;
			features.pDisabledValidationFeatures = nullptr;
			features.pEnabledValidationFeatures = enabled;
			features.pNext = instanceCreateInfo.pNext;
			instanceCreateInfo.pNext = &features;
			
		}
		return vkCreateInstance(&instanceCreateInfo, nullptr, &vkInstance);
	}

}