#include "Algorithm/Reduction.h"

namespace dyno
{
    template <typename T>
    Reduction<T>::Reduction() : Reduction(0) {
    }
    template <typename T>
    Reduction<T>::~Reduction() {
    }

    template <typename T>
    Reduction<T>::Reduction(uint num) {
        prepareBuffer(num);
        mReduceKernel = std::make_shared<VkProgram>(BUFFER(T), BUFFER(T), CONSTANT(int));
        mReduceKernel->load(getDynamicSpvFile<T>("shaders/glsl/core/Reduce.comp.spv"));
        mMaxKernel = std::make_shared<VkProgram>(BUFFER(T), BUFFER(T), CONSTANT(int), CONSTANT(T));
        mMaxKernel->load(getDynamicSpvFile<T>("shaders/glsl/core/Max.comp.spv"));
        mMinKernel = std::make_shared<VkProgram>(BUFFER(T), BUFFER(T), CONSTANT(int), CONSTANT(T));
        mMinKernel->load(getDynamicSpvFile<T>("shaders/glsl/core/Min.comp.spv"));
    }

    template <typename T>
    Reduction<T>* Reduction<T>::Create(uint n) {
        return new Reduction<T>(n);
    }

    template <typename T>
    void Reduction<T>::prepareBuffer(uint num) {
        m_num = num;
        mBuffer[0].resize(num);
        num = nextLevelSize(num);
        mBuffer[1].resize(num);
    }

    template <typename T>
    uint Reduction<T>::nextLevelSize(uint num) {
        return (num + blockSize) / blockSize;
    }

    template <typename T>
    T Reduction<T>::accumulate(Array<T, DeviceType::GPU> input) {
        return accumulate(input.begin(), input.size());
    }

    template <typename T>
    T Reduction<T>::maximum(Array<T, DeviceType::GPU> input) {
        return maximum(input.begin(), input.size());
    }

    template <typename T>
    T Reduction<T>::minimum(Array<T, DeviceType::GPU> input) {
        return minimum(input.begin(), input.size());
    }

    template <typename T>
    T Reduction<T>::accumulate(VkDeviceArray<T> input, uint num) {
        if (num != m_num) prepareBuffer(nextLevelSize(num));

        VkConstant<int> vk_num {(int)num};

        VkCompContext::current().push();

        mReduceKernel->begin();
        mReduceKernel->enqueue(vkDispatchSize(num, blockSize), &input, &mBuffer.front(), &vk_num);

        uint i = 0;
        for (uint level = mBuffer.front().size(); level > 1; level = nextLevelSize(level)) {
            vk_num.setValue(level);
            dim3 groupSize = vkDispatchSize(level, blockSize);

            auto& in_buf = mBuffer[i % mBuffer.size()];
            auto& out_buf = mBuffer[(++i) % mBuffer.size()];
            mReduceKernel->enqueue(groupSize, &in_buf, &out_buf, &vk_num);
        }

        mReduceKernel->end();

        auto fence = mReduceKernel->update(true);
        mReduceKernel->wait(fence.value());

        VkCompContext::current().pop();

        auto& out_buf = mBuffer[i % mBuffer.size()];
        std::vector<T> ret(1);
        vkTransfer(ret, 0, out_buf, 0, 1);

        T sum = ret[0];
        return sum;
    }

    template <typename T>
    T Reduction<T>::maximum(VkDeviceArray<T> input, uint num) {
        if (num != m_num) prepareBuffer(nextLevelSize(num));

        VkConstant<int> vk_num {(int)num};
        VkConstant<T> vk_init_val {std::numeric_limits<T>::min()};

        VkCompContext::current().push();
        mMaxKernel->begin();
        mMaxKernel->enqueue(vkDispatchSize(num, blockSize), &input, &mBuffer.front(), &vk_num, &vk_init_val);

        uint i = 0;
        for (uint level = mBuffer.front().size(); level > 1; level = nextLevelSize(level)) {
            vk_num.setValue(level);
            dim3 groupSize = vkDispatchSize(level, blockSize);

            auto& in_buf = mBuffer[i % mBuffer.size()];
            auto& out_buf = mBuffer[(++i) % mBuffer.size()];
            mMaxKernel->enqueue(groupSize, &in_buf, &out_buf, &vk_num, &vk_init_val);
        }

        mMaxKernel->end();

        auto fence = mMaxKernel->update(true);
        mMaxKernel->wait(fence.value());
        VkCompContext::current().pop();

        auto& out_buf = mBuffer[i % mBuffer.size()];
        std::vector<T> ret(1);
        vkTransfer(ret, 0, out_buf, 0, 1);

        T sum = ret[0];
        return sum;
    }

    template <typename T>
    T Reduction<T>::minimum(VkDeviceArray<T> input, uint num) {
        if (num != m_num) prepareBuffer(nextLevelSize(num));

        VkConstant<int> vk_num {(int)num};
        VkConstant<T> vk_init_val {std::numeric_limits<T>::max()};

        VkCompContext::current().push();
        mMinKernel->begin();
        mMinKernel->enqueue(vkDispatchSize(num, blockSize), &input, &mBuffer.front(), &vk_num, &vk_init_val);

        uint i = 0;
        for (uint level = mBuffer.front().size(); level > 1; level = nextLevelSize(level)) {
            vk_num.setValue(level);
            dim3 groupSize = vkDispatchSize(level, blockSize);

            auto& in_buf = mBuffer[i % mBuffer.size()];
            auto& out_buf = mBuffer[(++i) % mBuffer.size()];
            mMinKernel->enqueue(groupSize, &in_buf, &out_buf, &vk_num, &vk_init_val);
        }

        mMinKernel->end();

        auto fence = mMinKernel->update(true);
        mMinKernel->wait(fence.value());
        VkCompContext::current().pop();

        auto& out_buf = mBuffer[i % mBuffer.size()];
        std::vector<T> ret(1);
        vkTransfer(ret, 0, out_buf, 0, 1);

        T sum = ret[0];
        return sum;
    }

    template <typename T>
    T Reduction<T>::average(VkDeviceArray<T> val, uint num) {
        return accumulate(val, num) / num;
    }

    template class Reduction<int>;
    template class Reduction<float>;
    template class Reduction<double>;
    template class Reduction<uint>;

    Reduction<Vec3f>::Reduction() {
    }

    Reduction<Vec3f>::~Reduction() {
    }

    Vec3f Reduction<Vec3f>::accumulate(VkDeviceArray<Vec3f> input_device, uint num) {
        Vec3f ret;

        auto x_arr = m_arr_comp.get(input_device, 0);
        auto y_arr = m_arr_comp.get(input_device, 1);
        auto z_arr = m_arr_comp.get(input_device, 2);

        ret[0] = m_reduce_float.accumulate(x_arr);
        ret[1] = m_reduce_float.accumulate(y_arr);
        ret[2] = m_reduce_float.accumulate(z_arr);
        return ret;
    }

    Vec3f Reduction<Vec3f>::maximum(VkDeviceArray<Vec3f> input_device, uint num) {
        Vec3f ret;

        auto x_arr = m_arr_comp.get(input_device, 0);
        auto y_arr = m_arr_comp.get(input_device, 1);
        auto z_arr = m_arr_comp.get(input_device, 2);

        ret[0] = m_reduce_float.maximum(x_arr);
        ret[1] = m_reduce_float.maximum(y_arr);
        ret[2] = m_reduce_float.maximum(z_arr);
        return ret;
    }
    Vec3f Reduction<Vec3f>::minimum(VkDeviceArray<Vec3f> input_device, uint num) {
        Vec3f ret;

        auto x_arr = m_arr_comp.get(input_device, 0);
        auto y_arr = m_arr_comp.get(input_device, 1);
        auto z_arr = m_arr_comp.get(input_device, 2);

        ret[0] = m_reduce_float.minimum(x_arr);
        ret[1] = m_reduce_float.minimum(y_arr);
        ret[2] = m_reduce_float.minimum(z_arr);
        return ret;
    }
    Vec3f Reduction<Vec3f>::average(VkDeviceArray<Vec3f> val, uint num) {
        auto vec = accumulate(val, num);
        return vec / num;
    }

} // namespace dyno