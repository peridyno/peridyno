#include "VulkanBuffer.h"
#include <Vector.h>
#include <VkSystem.h>
#include <VkContext.h>

#include <glad/glad.h>

#include <iostream>


#ifdef WIN32
#include <handleapi.h>
#else
#include <unistd.h>
#endif

using namespace gl;

void createExternalVulkanBuffer(
    dyno::VkContext* ctx,
    VkDeviceSize size, 
    VkBufferUsageFlags usage, 
    VkMemoryPropertyFlags properties, 
    VkBuffer& buffer, 
    VkDeviceMemory& bufferMemory) {

    // OS platforms
#ifdef WIN32
    VkExternalMemoryHandleTypeFlags type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    VkExternalMemoryHandleTypeFlags type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkDevice device = ctx->deviceHandle();

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkExternalMemoryBufferCreateInfo externalInfo{};
    externalInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    externalInfo.handleTypes = type;
    bufferInfo.pNext = &externalInfo;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = ctx->getMemoryType(memRequirements.memoryTypeBits, properties);

    // enable export memory
    VkExportMemoryAllocateInfo memoryHandleEx{};
    memoryHandleEx.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    memoryHandleEx.handleTypes = type;
    allocInfo.pNext = &memoryHandleEx;  // <-- Enabling Export

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}


void gl::VulkanBuffer::create(int target, int usage)
{
    Buffer::create(target, usage);
    glCreateMemoryObjectsEXT(1, &memoryObject);
}

void gl::VulkanBuffer::release()
{
    Buffer::release();

#ifdef WIN32
    CloseHandle(handle);
#else
    if (fd != -1)
    {
        close(fd);
        fd = -1;
    }
#endif
    glDeleteMemoryObjectsEXT(1, &memoryObject);
}

void VulkanBuffer::allocate(int size) {

	std::cout << "allocate buffer: " << this->size << " -> " << size << " bytes" << std::endl;
	this->size = size;

	dyno::VkSystem*  vkSys = dyno::VkSystem::instance();
    dyno::VkContext* vkCtx = vkSys->currentContext();

    vkDestroyBuffer(vkCtx->deviceHandle(), buffer, 0);
    vkFreeMemory(vkCtx->deviceHandle(), memory, 0);

    createExternalVulkanBuffer(vkCtx, size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        buffer, memory);

    //
    VkDevice device = vkCtx->deviceHandle();

#ifdef WIN32
    VkMemoryGetWin32HandleInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    info.memory = memory;
    info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    auto vkGetMemoryWin32HandleKHR = 
        PFN_vkGetMemoryWin32HandleKHR(vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"));

    vkGetMemoryWin32HandleKHR(device, &info, &handle);
#else
    // TODO: for linux and other OS
#endif   
    
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, buffer, &req);

#ifdef WIN32
    glImportMemoryWin32HandleEXT(memoryObject, req.size, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, handle);
#else
    glImportMemoryFdEXT(bufGl.memoryObject, req.size, GL_HANDLE_TYPE_OPAQUE_FD_EXT, bufGl.fd);
    // fd got consumed
    bufGl.fd = -1;
#endif
    glNamedBufferStorageMemEXT(id, req.size, memoryObject, 0);

    glCheckError();
}

void VulkanBuffer::load(VkBuffer src, int size) {
	if (size > this->size) {
		this->allocate(size);
	}

    // copy
    dyno::VkContext* vkCtx = dyno::VkSystem::instance()->currentContext();

    if (copyCmd == VK_NULL_HANDLE) {
        copyCmd = vkCtx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    }

    // begin
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK_RESULT(vkBeginCommandBuffer(copyCmd, &beginInfo));

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0; // Optional
    copyRegion.dstOffset = 0; // Optional
    copyRegion.size = size;
    vkCmdCopyBuffer(copyCmd, src, buffer, 1, &copyRegion);

    // end and flush
    vkCtx->flushCommandBuffer(copyCmd, vkCtx->transferQueue, false);
}

