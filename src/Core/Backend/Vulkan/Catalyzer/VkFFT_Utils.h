#ifndef VKFFT_UTILS_H
#define VKFFT_UTILS_H
#include "vkFFT_Base.h"
#include <vector>
typedef struct {
	uint64_t X;
	uint64_t Y;
	uint64_t Z;
	uint64_t P;
	uint64_t B;
	uint64_t N;
	uint64_t R2C;
	uint64_t DCT;
} VkFFTUserSystemParameters;//an example structure used to pass user-defined system for benchmarking

#if(VKFFT_BACKEND==0)
VkResult CreateDebugUtilsMessengerEXT(VkGPU* vkGPU, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);
void DestroyDebugUtilsMessengerEXT(VkGPU* vkGPU, const VkAllocationCallbacks* pAllocator);
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);
VkResult setupDebugMessenger(VkGPU* vkGPU);
VkResult checkValidationLayerSupport();
std::vector<const char*> getRequiredExtensions(VkGPU* vkGPU, uint64_t sample_id);
VkResult createInstance(VkGPU* vkGPU, uint64_t sample_id);
VkResult findPhysicalDevice(VkGPU* vkGPU);
VkResult getComputeQueueFamilyIndex(VkGPU* vkGPU);
VkResult createDevice(VkGPU* vkGPU, uint64_t sample_id);
VkResult createFence(VkGPU* vkGPU);
VkResult createCommandPool(VkGPU* vkGPU);
VkFFTResult findMemoryType(VkGPU* vkGPU, uint64_t memoryTypeBits, uint64_t memorySize, VkMemoryPropertyFlags properties, uint32_t* memoryTypeIndex);
VkFFTResult allocateBuffer(VkGPU* vkGPU, VkBuffer* buffer, VkDeviceMemory* deviceMemory, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags propertyFlags, uint64_t size);
VkFFTResult transferDataFromCPU(VkGPU* vkGPU, void* arr, VkBuffer* buffer, uint64_t bufferSize);
VkFFTResult transferDataToCPU(VkGPU* vkGPU, void* arr, VkBuffer* buffer, uint64_t bufferSize);
#endif
VkFFTResult devices_list();
VkFFTResult performVulkanFFT(VkGPU* vkGPU, VkFFTApplication* app, VkFFTLaunchParams* launchParams, int inverse, uint64_t num_iter);
VkFFTResult performVulkanFFTiFFT(VkGPU* vkGPU, VkFFTApplication* app, VkFFTLaunchParams* launchParams, uint64_t num_iter, double* time_result);
#endif