#include "VkDescriptorCache.h"

#include <array>

#include "Helper/Hash.h"
#include "VulkanInitializers.hpp"
#include "VulkanTools.h"

bool operator<(const VkDescriptorSetLayoutBinding& a, const VkDescriptorSetLayoutBinding& b) {
    return a.binding == b.binding ? (a.descriptorType == b.descriptorType ? a.stageFlags < b.stageFlags
                                                                          : a.descriptorType < b.descriptorType

                                     )
                                  : a.binding < b.binding;
}

namespace
{
    std::size_t hashLayout(const VkDescriptorSetLayoutCreateInfo& info) {
        std::size_t seed {0};
        for (std::uint32_t i = 0; i < info.bindingCount; i++) {
            auto& bind = info.pBindings[i];
            dyno::hash_combine(seed, (((uint64_t)bind.binding) << 32) + bind.descriptorCount);
            dyno::hash_combine(seed, bind.descriptorType);
        }
        return seed;
    }
} // namespace

namespace dyno
{
    VkDescriptorCache::VkDescriptorCache(VkDevice device) : mDevice(device) {
        std::unique_lock lock {mMutex};
    }
    VkDescriptorCache::~VkDescriptorCache() {
        std::unique_lock lock {mMutex};

        for (auto& el : mPoolInfos) {
            el.second.reset(mDevice);
            vkDestroyDescriptorPool(mDevice, el.second.pool, nullptr);
        }
        for (auto& el : mLayerInfos) {
            vkDestroyDescriptorSetLayout(mDevice, el.second.layout, nullptr);
        }
    }

    std::optional<VkDescriptorSet> VkDescriptorCache::querySet(LayoutHandle key) {
        std::unique_lock lock {mMutex};
        // printf("layout: %d, set: %d\n", mLayerInfos.size(), mSetInfos.size());

        if (mLayerInfos.count(key)) {
            return mLayerInfos.at(key).query(*this);
        }
        return {};
    }
    void VkDescriptorCache::releaseSet(VkDescriptorSet set) {
        std::unique_lock lock {mMutex};
        // printf("layout: %d, set: %d\n", mLayerInfos.size(), mSetInfos.size());

        assert(mSetInfos.count(set));
        auto& setinfo = mSetInfos.at(set);
        auto& layerinfo = mLayerInfos.at(setinfo.layerKey);
        auto& poolinfo = mPoolInfos.at(setinfo.pool);
        setinfo.used = false;

        layerinfo.release(set);
    }

    std::optional<VkDescriptorSetLayout> VkDescriptorCache::layout(VkDescriptorSet set) const {
        LayoutHandle handle {0};
        {
            std::unique_lock lock {mMutex};
            if (mSetInfos.count(set)) {
                handle = mSetInfos.at(set).layerKey;
            }
        }
        return layout(handle);
    }

    std::optional<VkDescriptorSetLayout> VkDescriptorCache::layout(LayoutHandle handle) const {
        std::unique_lock lock {mMutex};
        if (mLayerInfos.count(handle)) {
            return mLayerInfos.at(handle).layout;
        }
        return {};
    }

    VkDescriptorCache::LayoutHandle VkDescriptorCache::queryLayoutHandle(const VkDescriptorSetLayoutCreateInfo& ci) {
        std::unique_lock lock {mMutex};
        auto key = hashLayout(ci);
        auto& info = layoutInfo(ci, key);
        info.ref++;
        return key;
    }

    void VkDescriptorCache::releaseLayoutHandle(LayoutHandle handle) {
        std::unique_lock lock {mMutex};
        if (mLayerInfos.count(handle)) {
            auto& info = mLayerInfos.at(handle);
            if (info.ref == 1 && info.used_count == 0) {
                // assert(info.used_count == 0);
                vkDestroyDescriptorSetLayout(mDevice, info.layout, nullptr);
                mLayerInfos.erase(handle);
                return;
            }
            assert(info.ref > 0);
            info.ref--;
        }
    }

    VkDescriptorCache::LayoutInfo& VkDescriptorCache::layoutInfo(const VkDescriptorSetLayoutCreateInfo& ci,
                                                                 LayoutHandle key) {
        if (!mLayerInfos.count(key)) {
            LayoutInfo info {};
            info.layerKey = key;
            VK_CHECK_RESULT(vkCreateDescriptorSetLayout(mDevice, &ci, nullptr, &info.layout));
            mLayerInfos.insert_or_assign(key, info);
        }
        return mLayerInfos.at(key);
    }

    VkDescriptorSet VkDescriptorCache::LayoutInfo::query(VkDescriptorCache& cache) {
        VkDescriptorSet out {VK_NULL_HANDLE};
        if (this->free.empty()) {
            auto set = cache.allocate(this->layout, this->layerKey);
            out = set;
        }
        else {
            out = this->free.back();
            this->free.pop_back();
        }
        this->used_count++;
        return out;
    }

    VkDescriptorCache::PoolInfo& VkDescriptorCache::queryPool() {
        const std::uint32_t PoolSize {100};
        for (auto& [handle, p] : mPoolInfos) {
            if (p.size < p.reserved) {
                return p;
            }
        }
        PoolInfo info {};
        info.reserved = PoolSize;
        {

            auto DPS = [=](VkDescriptorType type) -> VkDescriptorPoolSize { return VkDescriptorPoolSize {type, PoolSize}; };
            std::array sizes {DPS(VK_DESCRIPTOR_TYPE_SAMPLER),
                              DPS(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
                              DPS(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
                              DPS(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
                              DPS(VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER),
                              DPS(VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER),
                              DPS(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),
                              DPS(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
                              DPS(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC),
                              DPS(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC),
                              DPS(VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT)};

            VkDescriptorPoolCreateInfo descriptorPoolInfo =
                vks::initializers::descriptorPoolCreateInfo(sizes.size(), sizes.data(), PoolSize);
            VK_CHECK_RESULT(vkCreateDescriptorPool(mDevice, &descriptorPoolInfo, nullptr, &info.pool));
        }
        mPoolInfos.insert_or_assign(info.pool, info);
        return mPoolInfos.at(info.pool);
    }

    void VkDescriptorCache::LayoutInfo::release(VkDescriptorSet set) {
        this->free.push_back(set);
        this->used_count--;
    }

    VkDescriptorSet VkDescriptorCache::PoolInfo::allocate(VkDevice device, VkDescriptorSetLayout layout) {
        assert(this->size < this->reserved);

        VkDescriptorSet set {VK_NULL_HANDLE};
        VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(this->pool, &layout, 1);
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &set));
        this->size++;
        return set;
    }

    void VkDescriptorCache::PoolInfo::release(VkDevice device, VkDescriptorSet) {
        assert(this->size > 0);
        if (this->size > 0) this->size--;
        if (this->size == 0) {
            reset(device);
        }
    }

    void VkDescriptorCache::PoolInfo::reset(VkDevice device) {
        // assert(this->size == 0);
        VK_CHECK_RESULT(vkResetDescriptorPool(device, this->pool, 0));
        this->size = 0;
    }

    VkDescriptorSet VkDescriptorCache::allocate(VkDescriptorSetLayout layout, LayoutHandle key) {
        SetInfo info {};
        auto& poolInfo = queryPool();
        info.pool = poolInfo.pool;
        info.layerKey = key;
        info.used = true;
        auto set = poolInfo.allocate(mDevice, layout);
        mSetInfos.insert_or_assign(set, info);
        return set;
    }
} // namespace dyno