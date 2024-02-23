#include "VkCompContext.h"
#include "VkContext.h"
#include "VkSystem.h"
#include "Helper/Range.h"



namespace dyno
{
    struct VkCompContext::Frame {
		struct SubmitInfo {
			std::vector<VkCommandBuffer> cmds;
		};
        bool delaySubmit { false };
		std::vector<VkDescriptorSet> descriptors;
		std::unordered_map<std::intptr_t, VkSemaphore> semaphores;
		std::set<std::shared_ptr<vks::Buffer>> buffers;
        std::vector<VkFence> fences;
        std::vector<VkCommandBuffer> cmds;
        std::optional<VkCommandBuffer> activeCmd;
        std::vector<SubmitInfo> submits;
	};
    class VkCompContext::Private {
    public:
        Private(VkDevice device) : device(device), pool(VK_NULL_HANDLE), cmd(VK_NULL_HANDLE) {}
        ~Private() {}
        std::deque<Frame> stack;
        VkDevice device;
        VkCommandPool pool;
        VkCommandBuffer cmd;
        std::vector<VkCommandBuffer> freeCmds;
    };

    VkCompContext::VkCompContext(VkDevice device) : d_ptr(std::make_unique<Private>(device))
    {
    }
    VkCompContext::~VkCompContext() {
        reset();
    }


    bool VkCompContext::empty() const {
        DYNO_D(const VkCompContext);
        return d->stack.empty();
    }

    VkCompContext& VkCompContext::current() {
        thread_local VkCompContext comp{VkSystem::instance()->currentContext()->deviceHandle()};
        return comp;
    }

    void VkCompContext::flushAllSubmit() {
        DYNO_D(VkCompContext);
        for (auto& s : d->stack) {
            flushSubmit(s);
        }
    }

    void VkCompContext::flushSubmit(Frame& cur) {
        DYNO_D(VkCompContext);
        if (!cur.submits.empty()) {
		    auto ctx = VkSystem::instance()->currentContext();

			std::vector<VkSubmitInfo> infos;
			std::transform(cur.submits.begin(), cur.submits.end(), std::back_inserter(infos), [](const Frame::SubmitInfo& sinfo) {
				VkSubmitInfo out = vks::initializers::submitInfo();
				out.pCommandBuffers = sinfo.cmds.data();
				out.commandBufferCount = sinfo.cmds.size();
                return out;
			});
			VkFence fence;
			VkFenceCreateInfo fenceInfo = vks::initializers::fenceCreateInfo(VK_FLAGS_NONE);
			VK_CHECK_RESULT(vkCreateFence(d->device, &fenceInfo, nullptr, &fence));
			VkSystem::instance()->currentContext()->vkQueueSubmitSync(ctx->computeQueueHandle(), infos.size(), infos.data(), fence);
            cur.fences.emplace_back(fence);

            // clear
            for (auto& c : cur.submits) {
                d->freeCmds.insert(d->freeCmds.end(), c.cmds.begin(), c.cmds.end());
            }
            cur.submits.clear();
		}
        if (!cur.fences.empty()) {
            VK_CHECK_RESULT(vkWaitForFences(d->device, cur.fences.size(), cur.fences.data(), VK_TRUE, UINT64_MAX));
            for (auto f : cur.fences) {
                vkDestroyFence(d->device, f, nullptr);
            }
            cur.fences.clear();
        }
    }

    VkCompContext::Frame& VkCompContext::push() {
        DYNO_D(VkCompContext);
        d->stack.emplace_back(Frame {});
        return d->stack.back();
    }

    void VkCompContext::pop() {
        DYNO_D(VkCompContext);
        {
            auto& cur = d->stack.back();
            auto ctx = VkSystem::instance()->currentContext();
            auto& cache = ctx->descriptorCache();

            flushAllSubmit();

            for (auto set : cur.descriptors) {
                cache.releaseSet(set);
            }

            for (auto [_, sem] : cur.semaphores) {
                vkDestroySemaphore(d->device, sem, nullptr);
            }
            d->freeCmds.insert(d->freeCmds.end(), cur.cmds.begin(), cur.cmds.end());
        }
        d->stack.pop_back();
    }

    VkCompContext::Frame& VkCompContext::cur() {
        DYNO_D(VkCompContext);
        return d->stack.back();
    }

    const VkCompContext::Frame& VkCompContext::cur() const {
        DYNO_D(const VkCompContext);
        return d->stack.back();
    }

    VkDescriptorSet VkCompContext::allocDescriptorSet(VkCompContext::LayoutHandle handle) {
        auto& cache = VkSystem::instance()->currentContext()->descriptorCache();
        VkDescriptorSet set = cache.querySet(handle).value();
        cur().descriptors.push_back(set);
        return set;
    }

    VkCommandBuffer VkCompContext::commandBuffer(bool oneshot) {
        DYNO_D(VkCompContext);
        auto ctx = VkSystem::instance()->currentContext();
        if (d->pool == VK_NULL_HANDLE) {
            d->pool = ctx->createCommandPool(ctx->computeQueueFamilyIndex(), VkCommandPoolCreateFlagBits::VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
        }
        auto create_cmd = [ctx, d]() { 
            if (!d->freeCmds.empty()) {
                VkCommandBuffer cmd = d->freeCmds.back();
                d->freeCmds.pop_back();
                return cmd;
            }
            return ctx->createCommandBuffer(VkCommandBufferLevel::VK_COMMAND_BUFFER_LEVEL_PRIMARY, d->pool, false); 
        };
        if(oneshot) {
            return create_cmd();
        }
        else {
            if (empty()) {
                if (d->cmd == VK_NULL_HANDLE) {
                    d->cmd = create_cmd();
                }
                return d->cmd;
            }
            else {
                auto& c = cur();
                if (!c.activeCmd)
                    c.activeCmd = create_cmd();
                return c.activeCmd.value();
            }
        }
    }

    void VkCompContext::releaseCommandBuffer(VkCommandBuffer buf) {
        DYNO_D(VkCompContext);
        d->freeCmds.emplace_back(buf);
    }

    void VkCompContext::reset() {
        DYNO_D(VkCompContext);
        assert(d->stack.empty());
        if (d->cmd != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(d->device, d->pool, 1, &d->cmd);
            d->cmd = VK_NULL_HANDLE;
        }
        if (d->freeCmds.size()) {
            vkFreeCommandBuffers(d->device, d->pool, d->freeCmds.size(), d->freeCmds.data());
            d->freeCmds.clear();
        }
        if (d->pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(d->device, d->pool, nullptr);
            d->pool = VK_NULL_HANDLE;
        }
    }

    auto VkCompContext::registerSemaphore(std::intptr_t handle) -> VkSemaphore {
        DYNO_D(VkCompContext);
        VkSemaphore out {VK_NULL_HANDLE};
        auto& c = cur();
		VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
        VK_CHECK_RESULT(vkCreateSemaphore(d->device, &semaphoreCreateInfo, nullptr, &out));
        assert(c.semaphores.count(handle) == 0);
        c.semaphores.insert({ handle, out });
        return out;
    }
    auto VkCompContext::consumeSemaphore(std::intptr_t handle) -> std::optional<VkSemaphore> {
        DYNO_D(VkCompContext);
        for (auto& cur : dyno::reverse(d->stack)) {
            if (cur.semaphores.count(handle)) {
                auto out = cur.semaphores.at(handle);
                cur.semaphores.erase(handle);
                return out;
            }
        }
        return {};
    }
    void VkCompContext::delaySubmit(bool val) {
        cur().delaySubmit = val;
    }

    void VkCompContext::submit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence) {
        DYNO_D(VkCompContext);
        auto& c = cur();
        if(c.delaySubmit) {
            for (uint32_t i = 0; i < submitCount; i++) {
                Frame::SubmitInfo info;
                auto begin = pSubmits[i].pCommandBuffers;
                auto count = pSubmits[i].commandBufferCount;
                info.cmds.insert(info.cmds.end(), begin, begin + count);
                assert(fence == VK_NULL_HANDLE);
                c.submits.emplace_back(info);
            }
        }
        else {
            if (fence == VK_NULL_HANDLE) {
                VkFenceCreateInfo fenceInfo = vks::initializers::fenceCreateInfo(VK_FLAGS_NONE);
                VK_CHECK_RESULT(vkCreateFence(d->device, &fenceInfo, nullptr, &fence));
                c.fences.emplace_back(fence);
            }
            else {
                flushAllSubmit();
            }
            VkSystem::instance()->currentContext()->vkQueueSubmitSync(queue, submitCount, pSubmits, fence);
            for (uint32_t i = 0; i < submitCount; i++) {
                auto begin = pSubmits[i].pCommandBuffers;
                auto count = pSubmits[i].commandBufferCount;
                c.cmds.insert(c.cmds.end(), begin, begin + count);
            }
        }
        for (uint32_t i = 0; i < submitCount; i++) {
			auto begin = pSubmits[i].pCommandBuffers;
			auto end = begin + pSubmits[i].commandBufferCount;
			if (end != std::find_if(begin, end,[&c](auto cmd) {
				return std::optional{ cmd } == c.activeCmd;
				})) {
				c.activeCmd = {};

                return;
			}
        }
    }


    void VkCompContext::registerBuffer(std::shared_ptr<vks::Buffer> buf) {
        if (!empty()) {
            cur().buffers.insert(buf);
        }
    }

    void VkCompContext::addTransferBarrier(VkCommandBuffer commandBuffer, VkBuffer buffer) {
        auto ctx = VkSystem::instance()->currentContext();
		VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
        bufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        bufferBarrier.srcQueueFamilyIndex = ctx->computeQueueFamilyIndex();
        bufferBarrier.dstQueueFamilyIndex = ctx->computeQueueFamilyIndex();
		bufferBarrier.size = VK_WHOLE_SIZE;
        bufferBarrier.buffer = buffer;

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_FLAGS_NONE,
			0, nullptr,
			1, &bufferBarrier,
			0, nullptr);
    }


    bool VkCompContext::isDelaySubmit() const {
        return empty() ? false : cur().delaySubmit;
    }

    VkCompContext::Holder::Holder() {
        VkCompContext::current().push();
    }
    VkCompContext::Holder::~Holder() {
        VkCompContext::current().pop();
    }
    void VkCompContext::Holder::delaySubmit(bool v) {
        VkCompContext::current().delaySubmit(v);
    }
} // namespace dyno