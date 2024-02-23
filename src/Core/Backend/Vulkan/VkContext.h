#pragma once

#include <algorithm>
#include <assert.h>
#include <exception>
#include <map>
#include <mutex>

#include "Platform.h"
#include "VulkanTools.h"
#include "VulkanBuffer.h"
#include "VkDescriptorCache.h"

VK_DEFINE_HANDLE(VmaAllocator)
VK_DEFINE_HANDLE(VmaPool)

namespace dyno
{
    class VkSystem;

    class VkContext {
        friend class VkSystem;

    public:
        enum MemPoolType
        {
            DevicePool,
            HostPool,
            UniformPool,
            EndType
        };

        explicit VkContext(VkPhysicalDevice physicalDevice);
        ~VkContext();

        void createPipelineCache();
        VkResult createLogicalDevice(VkPhysicalDeviceFeatures enabledFeatures,
                                     std::vector<const char*> enabledExtensions, void* pNextChain,
                                     bool useSwapChain = true,
                                     VkQueueFlags requestedQueueTypes = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);

        bool useMemPool() const;
        bool enableDebugMarkers() const;
        VkDevice deviceHandle() const;
        VkCommandPool commandPool() const;
        VkQueue graphicsQueueHandle() const;
        VkQueue computeQueueHandle() const;
        VkQueue transferQueueHandle() const;

        uint32_t graphicsQueueFamilyIndex() const;
        uint32_t computeQueueFamilyIndex() const;
        uint32_t transferQueueFamilyIndex() const;

        const VkPhysicalDeviceProperties& properties() const;
        const VkPhysicalDeviceFeatures& enabledFeatures() const;

        VkPhysicalDevice physicalDeviceHandle() const;
        VkPipelineCache pipelineCacheHandle() const;

        // Check whether the compute queue family is distinct from the graphics queue family
        bool isComputeQueueSpecial();

        VkDescriptorCache& descriptorCache();

        uint32_t getMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties,
                               VkBool32* memTypeFound = nullptr) const;
        VkResult createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags,
                              VkDeviceSize size, VkBuffer* buffer, VkDeviceMemory* memory, void* data = nullptr);
        VkResult createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags,
                              std::shared_ptr<vks::Buffer>& buffer, VkDeviceSize size, const void* data = nullptr);
        VkResult createBuffer(MemPoolType poolType, std::shared_ptr<vks::Buffer>& buffer, const void* data = nullptr);
        void copyBuffer(vks::Buffer* src, vks::Buffer* dst, VkQueue queue, VkBufferCopy* copyRegion = nullptr);
        VkCommandPool createCommandPool(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags createFlags =
                                                                       VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
        VkCommandBuffer createCommandBuffer(VkCommandBufferLevel level, VkCommandPool pool, bool begin = false);
        VkCommandBuffer createCommandBuffer(VkCommandBufferLevel level, bool begin = false);
        void flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, VkCommandPool pool, bool free = true);
        void flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, bool free = true);
        bool extensionSupported(std::string extension);
        VkFormat getSupportedDepthFormat(bool checkSamplingSupport);
        void vkQueueSubmitSync(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence);
        void destroy(VkCommandPool, VkCommandBuffer);

    private:
        uint32_t getQueueFamilyIndex(VkQueueFlagBits queueFlags) const;
        VkResult createMemoryPool(VkInstance instance, uint32_t apiVerion);

        class Private;
        DYNO_DECLARE_PRIVATE(Private);
        std::unique_ptr<Private> d_ptr;
    };
} // namespace dyno