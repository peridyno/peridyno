#pragma once
#include "Array/Array.h"
#include "STL/List.h"
#include "VkConstant.h"
#include "VkProgram.h"

namespace dyno
{
    class VkListAllocator {
    public:
        struct Push {
            uint32_t num_list;
            uint32_t num_element;
            uint32_t element_size;
            VkDeviceAddress lists;
            VkDeviceAddress indexs;
            VkDeviceAddress elements;

            struct List {
                uint32_t max_size;
                uint64_t address;
                uint64_t size;
            };
        };

        VkListAllocator();
        ~VkListAllocator();

        template <typename T>
        void allocate(DArray<List<T>> lists, DArray<T> elements, DArray<uint32_t> indexs) {
            static_assert(sizeof(List<T>) == sizeof(Push::List));
            VkConstant<Push> push;
            push->num_list = lists.size();
            push->num_element = elements.size();
            push->element_size = sizeof(T);
            push->lists = lists.handle()->bufferAddress();
            push->indexs = indexs.handle()->bufferAddress();
            push->elements = elements.handle()->bufferAddress();

            mKernel->flush(vkDispatchSize(push->num_element, 64), &push);
        }

    private:
        std::shared_ptr<VkProgram> mKernel;
    };
} // namespace dyno