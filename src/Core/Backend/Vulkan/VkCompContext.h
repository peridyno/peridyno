#pragma once

#include <optional>
#include <deque>
#include <vector>
#include <unordered_map>

#include "VkVariable.h"
#include "Platform.h"

namespace dyno
{
    class VkCompContext {
    public:
        using LayoutHandle = std::size_t;

        struct Holder {
            Holder();
            ~Holder();

            void delaySubmit(bool);
        };

        static VkCompContext& current();
        VkCompContext(VkDevice);
        ~VkCompContext();

        struct Frame;
        
        Frame& push();
        void pop();
        Frame& cur();
        const Frame& cur() const;

        bool empty() const;

        auto allocDescriptorSet(LayoutHandle) -> VkDescriptorSet;

        auto commandBuffer(bool oneshot = false) -> VkCommandBuffer;
        void releaseCommandBuffer(VkCommandBuffer);

        bool isDelaySubmit() const;
        void delaySubmit(bool);

        auto registerSemaphore(std::intptr_t handle) -> VkSemaphore;
        auto consumeSemaphore(std::intptr_t handle) -> std::optional<VkSemaphore>; 

        void registerBuffer(std::shared_ptr<vks::Buffer>);

        void addTransferBarrier(VkCommandBuffer commandBuffer, VkBuffer buffer);

		void submit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence);
        void reset();
    private:
        void flushAllSubmit();
        void flushSubmit(Frame&);

        class Private;
        DYNO_DECLARE_PRIVATE(Private);
        std::unique_ptr<Private> d_ptr;
    };
} // namespace dyno