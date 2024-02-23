#pragma once

#include <unordered_map>
#include <set>
#include <mutex>
#include <optional>

#include "Platform.h"
#include "vulkan/vulkan.h"

bool operator<(const VkDescriptorSetLayoutBinding&, const VkDescriptorSetLayoutBinding&);

namespace dyno
{
    class VkDescriptorCache {
    public:
        VkDescriptorCache(VkDevice device);
        ~VkDescriptorCache();
        using LayoutHandle = std::size_t;
        static constexpr LayoutHandle NullHandle {0};

        std::optional<VkDescriptorSet> querySet(LayoutHandle);
        void releaseSet(VkDescriptorSet);

        std::optional<VkDescriptorSetLayout> layout(VkDescriptorSet) const;
        std::optional<VkDescriptorSetLayout> layout(LayoutHandle) const;

        LayoutHandle queryLayoutHandle(const VkDescriptorSetLayoutCreateInfo&);
        void releaseLayoutHandle(LayoutHandle);

    private:
        struct SetInfo {
            bool used {false};
            VkDescriptorPool pool {VK_NULL_HANDLE};
            LayoutHandle layerKey {NullHandle};
        };
        struct LayoutInfo {
            std::size_t ref {0};
            LayoutHandle layerKey {NullHandle};
            VkDescriptorSetLayout layout {VK_NULL_HANDLE};
            std::size_t used_count {0};
            std::vector<VkDescriptorSet> free;

            VkDescriptorSet query(VkDescriptorCache&);
            void release(VkDescriptorSet);
        };
        struct PoolInfo {
            VkDescriptorPool pool {VK_NULL_HANDLE};
            std::size_t size {0};
            std::size_t reserved {0};

            VkDescriptorSet allocate(VkDevice, VkDescriptorSetLayout);
            void release(VkDevice, VkDescriptorSet);
            void reset(VkDevice);
        };

        VkDescriptorSet allocate(VkDescriptorSetLayout, LayoutHandle);
        PoolInfo& queryPool();
        LayoutInfo& layoutInfo(const VkDescriptorSetLayoutCreateInfo&, LayoutHandle);

        VkDevice mDevice;

        std::unordered_map<VkDescriptorSet, SetInfo> mSetInfos;
        std::unordered_map<LayoutHandle, LayoutInfo> mLayerInfos;
        std::unordered_map<VkDescriptorPool, PoolInfo> mPoolInfos;
        mutable std::mutex mMutex;
    };

} // namespace dyno
