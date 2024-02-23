#pragma once

#include <array>

#include "Vector.h"
#include "Catalyzer/VkReduce.h"
#include "Array/ArrayComponent.h"
#include "Array/Array.h"

namespace dyno
{
    template <typename T>
    class Reduction {
    public:
        static constexpr int blockSize {256};

        Reduction();
        ~Reduction();

        static Reduction* Create(uint n);

        T accumulate(Array<T, DeviceType::GPU>);
        T maximum(Array<T, DeviceType::GPU>);
        T minimum(Array<T, DeviceType::GPU>);

        T accumulate(VkDeviceArray<T> val, uint num);
        T maximum(VkDeviceArray<T> val, uint num);
        T minimum(VkDeviceArray<T> val, uint num);
        T average(VkDeviceArray<T> val, uint num);

    private:
        Reduction(uint num);
        void prepareBuffer(uint num);
        uint nextLevelSize(uint num);

        uint m_num;

        std::array<VkDeviceArray<T>, 2> mBuffer;
        std::shared_ptr<VkProgram> mReduceKernel;
        std::shared_ptr<VkProgram> mMaxKernel;
        std::shared_ptr<VkProgram> mMinKernel;
    };

    template <>
    class Reduction<Vec3f> {
    public:
        static constexpr auto vecSize = sizeof(Vec3f) / sizeof(float);
        Reduction();

        static Reduction* Create(uint n);
        ~Reduction();

        Vec3f accumulate(VkDeviceArray<Vec3f> val, uint num);
        Vec3f maximum(VkDeviceArray<Vec3f> val, uint num);
        Vec3f minimum(VkDeviceArray<Vec3f> val, uint num);
        Vec3f average(VkDeviceArray<Vec3f> val, uint num);

    private:
        Reduction<float> m_reduce_float;
        ArrayComponet<float> m_arr_comp;
    };

    template <>
    class Reduction<Vec3d> {
    public:
        Reduction();

        static Reduction* Create(uint n);
        ~Reduction();

        Vec3d accumulate(VkDeviceArray<Vec3d> val, uint num);
        Vec3d maximum(VkDeviceArray<Vec3d> val, uint num);
        Vec3d minimum(VkDeviceArray<Vec3d> val, uint num);
        Vec3d average(VkDeviceArray<Vec3d> val, uint num);

    private:
        Reduction<double> m_reduce_double;
        ArrayComponet<double> m_arr_comp;
    };
} // namespace dyno
