#pragma once
#include "VkDeviceArray.h"
#include "VkProgram.h"
#include "Array/Array.h"
#include "Vector/VectorBase.h"

namespace dyno
{

    template <typename T>
    class ArrayComponet {
    public:
        ArrayComponet();
        ~ArrayComponet();
        template <typename Vt, int Dim>
        Array<T, DeviceType::GPU> get(VkDeviceArray<Vector<Vt, Dim>> input, int offset);

        Array<T, DeviceType::GPU> get(VkDeviceArray<T> input, int stride, int offset);

    private:
        std::shared_ptr<VkProgram> mArrCompKernel;
    };
} // namespace dyno

namespace dyno
{
    template <typename T>
    ArrayComponet<T>::ArrayComponet()
        : mArrCompKernel(
              std::make_shared<VkProgram>(BUFFER(T), BUFFER(T), CONSTANT(int), CONSTANT(int), CONSTANT(int))) {
        mArrCompKernel->load(getDynamicSpvFile<T>("shaders/glsl/array/ArrayComponent.comp.spv"));
    }

    template <typename T>
    ArrayComponet<T>::~ArrayComponet() {
    }

    template <typename T>
    template <typename Vt, int Dim>
    Array<T, DeviceType::GPU> ArrayComponet<T>::get(VkDeviceArray<Vector<Vt, Dim>> input, int offset) {
        int stride = sizeof(Vector<Vt, Dim>);
        Array<T, DeviceType::GPU> out(input.size());
        VkConstant<int> vk_num {(int)out.size()}, vk_stride {stride}, vk_offset {offset};
        mArrCompKernel->flush(vkDispatchSize(input.size(), 256), &input, out.handle(), &vk_num, &vk_stride, &vk_offset);
        return out;
    }

    template <typename T>
    Array<T, DeviceType::GPU> ArrayComponet<T>::get(VkDeviceArray<T> input, int stride, int offset) {
        Array<T, DeviceType::GPU> out(input.size());
        VkConstant<int> vk_num {(int)out.size()}, vk_stride {stride}, vk_offset {offset};
        mArrCompKernel->flush(vkDispatchSize(input.size(), 256), &input, out.handle(), &vk_num, &vk_stride, &vk_offset);
        return out;
    }
} // namespace dyno