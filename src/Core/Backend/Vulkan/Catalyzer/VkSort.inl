
#include "VkTransfer.h"

namespace dyno
{
    struct VkSortPush {
        uint n;
        uint h;
        uint algorithm;
        uint direction;
    };

    inline int ComputeAddZeroSize(int OriginalSize) {
        int AddZeroSize = 2;
        while (AddZeroSize < OriginalSize) {
            AddZeroSize *= 2;
        }

        return AddZeroSize;
    }

    template <typename T>
	void VkSort<T>::sort(CArray<T>& data, uint32_t SortType) {
        DArray<T> Ddata;
        Ddata.assign(data);
        sort(*Ddata.handle(), SortType);
        data.assign(Ddata);
        Ddata.clear();
    }

    template <typename T>
	void VkSort<T>::sort(DArray<T> data, uint32_t SortType) {
        sort(*data.handle(), SortType);
    }

    template <typename T>
    void VkSort<T>::sort(VkDeviceArray<T> data, uint32_t SortType) {
        uint32_t globalSize = ComputeAddZeroSize(data.size());
        uint32_t workgroup_size_x = 64;
        const uint32_t workgroup_count = globalSize / (workgroup_size_x * 2);

        uint32_t h = workgroup_size_x * 2;
        dim3 groupSize;
        groupSize.x = workgroup_count;

        VkCompContext::current().push();
        mSortKernel->begin();

        VkConstant<VkSortPush> push;
        push->n = data.size();
        push->h = h;
        push->algorithm = SortParam::eLocalBms;
        // SortType UP 0/ DOWN 1
        push->direction = SortType;
        mSortKernel->enqueue(groupSize, &data, &push);
        // we must now double h, as this happens before every flip
        h *= 2;

        for (; h <= globalSize; h *= 2) {
            push->h = h;
            push->algorithm = SortParam::eBigFlip;
            mSortKernel->enqueue(groupSize, &data, &push);

            for (uint32_t hh = h / 2; hh > 1; hh /= 2) {
                if (hh <= workgroup_size_x * 2) {
                    push->h = hh;
                    push->algorithm = SortParam::eLocalDisperse;
                    mSortKernel->enqueue(groupSize, &data, &push);
                    break;
                }
                else {
                    push->h = hh;
                    push->algorithm = SortParam::eBigDisperse;
                    mSortKernel->enqueue(groupSize, &data, &push);
                }
            }
        }
        mSortKernel->end();
        mSortKernel->update(true);
        mSortKernel->wait();

        VkCompContext::current().pop();
    }



    template <typename T>
    VkSort<T>::VkSort(std::filesystem::path spv): mSortKernel(std::make_shared<VkProgram>(BUFFER(T), CONSTANT(VkSortPush))){
        if(spv.empty())
            mSortKernel->load(getDynamicSpvFile<T>("shaders/glsl/core/Sort.comp.spv"));
        else
            mSortKernel->load(spv.string());
    }

    template <typename T>
    VkSort<T>::~VkSort() {
        mSortKernel = nullptr;
    }


    template <typename TKey, typename TVal>
    VkSortByKey<TKey, TVal>::VkSortByKey(std::filesystem::path spv) : mSortKernel(std::make_shared<VkProgram>(BUFFER(TKey), BUFFER(TVal), CONSTANT(VkSortPush))) {
        if(spv.empty()) {
            mSortKernel->load(getDynamicSpvFile<TKey>("shaders/glsl/core/SortByKey.comp.spv"));
        } else {
            mSortKernel->load(spv.string());
        }
    }

    template <typename TKey, typename TVal>
    VkSortByKey<TKey, TVal>::~VkSortByKey() {
    }

    template <typename TKey, typename TVal>
    void VkSortByKey<TKey, TVal>::sortByKey(CArray<TKey>& keys, CArray<TVal>& values, uint32_t SortType) {
        DArray<TKey> dkeys, dvalues;
        dkeys.assign(keys);
        dvalues.assign(values);

        sortByKey(dkeys, dvalues, SortType);

        keys.assign(dkeys);
        values.assign(dvalues);
    }

    template <typename TKey, typename TVal>
	void VkSortByKey<TKey, TVal>::sortByKey(DArray<TKey> key, DArray<TVal> val, uint32_t SortType) {
	    sortByKey(*key.handle(), *val.handle(), SortType);
    }

    template <typename TKey, typename TVal>
    void VkSortByKey<TKey, TVal>::sortByKey(VkDeviceArray<TKey> key, VkDeviceArray<TVal> val, uint32_t SortType) {
        assert(key.size() <= val.size());

        VkCompContext::Holder holder;

        uint32_t globalSize = ComputeAddZeroSize(key.size());
        uint32_t workgroup_size_x = 64;
        const uint32_t workgroup_count = std::max<uint32_t>(1, globalSize / (workgroup_size_x * 2));

        uint32_t h = workgroup_size_x * 2;
        dim3 groupSize;
        groupSize.x = workgroup_count;

        mSortKernel->begin();

        VkConstant<VkSortPush> push;
        push->n = key.size();
        push->h = h;
        push->algorithm = SortParam::eLocalBms;
        // SortType UP 0/ DOWN 1
        push->direction = SortType;
        mSortKernel->enqueue(groupSize, &key, &val, &push);
        // we must now double h, as this happens before every flip
        h *= 2;

        for (; h <= globalSize; h *= 2) {
            push->h = h;
            push->algorithm = SortParam::eBigFlip;
            mSortKernel->enqueue(groupSize, &key, &val, &push);

            for (uint32_t hh = h / 2; hh > 1; hh /= 2) {
                if (hh <= workgroup_size_x * 2) {
                    push->h = hh;
                    push->algorithm = SortParam::eLocalDisperse;
                    mSortKernel->enqueue(groupSize, &key, &val, &push);
                    break;
                }
                else {
                    push->h = hh;
                    push->algorithm = SortParam::eBigDisperse;
                    mSortKernel->enqueue(groupSize, &key, &val, &push);
                }
            }
        }
        mSortKernel->end();
        mSortKernel->update(true);
        mSortKernel->wait();
    }
} // namespace dyno
