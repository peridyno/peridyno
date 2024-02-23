#include "VkDeviceArray.h"
#include "VkUniform.h"
#include "Array/ArrayTools.h"

#include "Catalyzer/VkScan.h"
#include "Catalyzer/VkReduce.h"

namespace dyno
{
    struct ArrayListInfo {
        uint32_t arraySize = 0;
        uint32_t totalSize = 0;
    };

    template <typename T>
    class ArrayList<T, DeviceType::GPU> {
    public:
        using value_t = T;

        ArrayList();

        /*!
         *	\brief	Should not release data here, call Release() explicitly.
         */
        ~ArrayList() {};

        void resize(const std::vector<uint32_t>& num);
        void resize(const DArray<uint32_t>& num);
        void resize(const VkDeviceArray<uint32_t>& num);

        void assign(const ArrayList<T, DeviceType::GPU>& src);
        void assign(const ArrayList<T, DeviceType::CPU>& src);
        void assign(const std::vector<std::vector<T>>& src);

        uint32_t size() {
            return mIndex.size();
        }
        inline uint elementSize() const {
            return mElements.size();
        }

        const DArray<uint32_t>& index() const {
            return mIndex;
        }
        const DArray<T>& elements() const {
            return mElements;
        }
        const DArray<List<T>>& lists() const {
            return mLists;
        }

        inline bool isCPU() const {
            return false;
        }
        inline bool isGPU() const {
            return true;
        }
        inline bool isEmpty() const {
            return mIndex.size() == 0;
        }

        void clear();

    public:
        DArray<uint> mIndex;
        DArray<T> mElements;

        DArray<List<T>> mLists;

        VkUniform<ArrayListInfo> mInfo;

        VkListAllocator mAllocator;
    };

    template <typename T>
    using DArrayList = ArrayList<T, DeviceType::GPU>;

    template <typename T>
    void ArrayList<T, DeviceType::CPU>::assign(const ArrayList<T, DeviceType::GPU>& src) {
        mIndex.assign(src.index());
        mElements.assign(src.elements());

        mLists.resize(mIndex.size());

        // redirect the element address
        for (int i = 0; i < mIndex.size(); i++) {
            // mLists[i].reserve(mElements.begin() + mIndex[i], this->size(i));
            uint num = size(i);
            mLists[i].assign(mElements.begin() + mIndex[i], num, num);
        }
    }

    template <typename T>
    ArrayList<T, DeviceType::GPU>::ArrayList() {
    }

    template <typename T>
    void ArrayList<T, DeviceType::GPU>::resize(const std::vector<uint32_t>& num) {
        DArray<uint32_t> deviceNum;
        deviceNum.resize(num.size());

        vkTransfer(*deviceNum.handle(), num);

        this->resize(deviceNum);

        deviceNum.clear();
    }

    template <typename T>
    void ArrayList<T, DeviceType::GPU>::resize(const DArray<uint32_t>& num) {
        this->resize(*num.handle());
    }

    template <typename T>
    void ArrayList<T, DeviceType::GPU>::resize(const VkDeviceArray<uint32_t>& num) {
        mIndex.resize(num.size());
        mLists.resize(num.size());

        VkScan<uint32_t> mScan;
        VkReduce<uint32_t> mReduce;

        mScan.scan(*mIndex.handle(), num, EXCLUSIVESCAN);

        uint32_t totalNum = mReduce.reduce(num);

        mElements.resize(totalNum);

        ArrayListInfo info;
        info.arraySize = num.size();
        info.totalSize = totalNum;

        mInfo.setValue(info);

        mAllocator.allocate(mLists, mElements, mIndex);
    }

    template <typename T>
    void ArrayList<T, DeviceType::GPU>::assign(const ArrayList<T, DeviceType::CPU>& src) {
        mIndex.assign(src.index());
        mElements.assign(src.elements());
        mLists.assign(src.lists());
        mAllocator.allocate(mLists, mElements, mIndex);
    }

    template <typename T>
    void ArrayList<T, DeviceType::GPU>::assign(const ArrayList<T, DeviceType::GPU>& src) {
        mIndex.assign(src.index());
        mElements.assign(src.elements());
        mLists.assign(src.lists());
        mAllocator.allocate(mLists, mElements, mIndex);
    }

    template <typename T>
    void ArrayList<T, DeviceType::GPU>::assign(const std::vector<std::vector<T>>& src) {
        size_t indNum = src.size();
        CArray<uint> hIndex(indNum);
        CArray<T> hElements;

        size_t eleNum = 0;
        for (int i = 0; i < src.size(); i++) {
            hIndex[i] = (uint)eleNum;
            eleNum += src[i].size();

            for (int j = 0; j < src[i].size(); j++) {
                hElements.pushBack(src[i][j]);
            }
        }

        CArray<List<T>> lists;
        lists.resize(indNum);

        mIndex.assign(hIndex);
        mElements.assign(hElements);
        T* stAdr = mElements.begin();

        eleNum = 0;
        for (int i = 0; i < src.size(); i++) {
            size_t num_i = src[i].size();
            List<T> lst;
            lst.assign(stAdr + eleNum, num_i, num_i);
            lists[i] = lst;

            eleNum += src[i].size();
        }

        mLists.assign(lists);
    }

    template <typename T>
    void ArrayList<T, DeviceType::GPU>::clear() {
        mIndex.clear();
        mElements.clear();
		mLists.clear();
    }

} // namespace dyno
