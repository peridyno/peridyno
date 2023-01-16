#include "ArrayTools.h"
#include "Algorithm/Scan.h"
#include "Algorithm/Reduction.h"

namespace dyno
{
	template<class ElementType>
	class ArrayList<ElementType, DeviceType::GPU>
	{
	public:
		ArrayList()
		{
		};

		/*!
		*	\brief	Do not release memory here, call clear() explicitly.
		*/
		~ArrayList() {};

		/**
		 * @brief Pre-allocate GPU space for
		 *
		 * @param counts
		 * @return true
		 * @return false
		 */
		bool resize(const DArray<uint>& counts);
		bool resize(const uint arraySize, const uint eleSize);


		bool resize(uint num);

		template<typename ET2>
		bool resize(const ArrayList<ET2, DeviceType::GPU>& src);

		DYN_FUNC inline uint size() const { return mLists.size(); }
		DYN_FUNC inline uint elementSize() const { return mElements.size(); }

		GPU_FUNC inline List<ElementType>& operator [] (unsigned int id) {
			return mLists[id];
		}

		GPU_FUNC inline List<ElementType>& operator [] (unsigned int id) const {
			return mLists[id];
		}

		DYN_FUNC inline bool isCPU() const { return false; }
		DYN_FUNC inline bool isGPU() const { return true; }
		DYN_FUNC inline bool isEmpty() const { return mIndex.size() == 0; }

		void clear();

		void assign(const ArrayList<ElementType, DeviceType::GPU>& src);
		void assign(const ArrayList<ElementType, DeviceType::CPU>& src);
		void assign(const std::vector<std::vector<ElementType>>& src);

		friend std::ostream& operator<<(std::ostream& out, const ArrayList<ElementType, DeviceType::GPU>& aList)
		{
			ArrayList<ElementType, DeviceType::CPU> hList;
			hList.assign(aList);
			out << hList;

			return out;
		}

		const DArray<uint>& index() const { return mIndex; }
		const DArray<ElementType>& elements() const { return mElements; }
		const DArray<List<ElementType>>& lists() const { return mLists; }

		/*!
		*	\brief	To avoid erroneous shallow copy.
		*/
		ArrayList<ElementType, DeviceType::GPU>& operator=(const ArrayList<ElementType, DeviceType::GPU>&) = delete;

	private:
		DArray<uint> mIndex;
		DArray<ElementType> mElements;

		DArray<List<ElementType>> mLists;
	};

	template<typename ElementType>
	using DArrayList = ArrayList<ElementType, DeviceType::GPU>;

	template<class ElementType>
	void ArrayList<ElementType, DeviceType::CPU>::assign(const ArrayList<ElementType, DeviceType::GPU>& src)
	{
		mIndex.assign(src.index());
		mElements.assign(src.elements());

		mLists.assign(src.lists());

		//redirect the element address
		for (int i = 0; i < src.size(); i++)
		{
			mLists[i].reserve(mElements.begin() + mIndex[i], mLists[i].size());
		}
	}

	template<class ElementType>
	void ArrayList<ElementType, DeviceType::GPU>::clear()
	{
		mIndex.clear();
		mElements.clear();
		mLists.clear();
	}

	template<class ElementType>
	bool ArrayList<ElementType, DeviceType::GPU>::resize(const DArray<uint>& counts)
	{
		assert(counts.size() > 0);

		if (mIndex.size() != counts.size())
		{
			mIndex.resize(counts.size());
			mLists.resize(counts.size());
		}

		mIndex.assign(counts);

		Reduction<uint> reduce;
		uint total_num = reduce.accumulate(mIndex.begin(), mIndex.size());

		Scan<uint> scan;
		scan.exclusive(mIndex);

		//printf("total num 2 = %d\n", total_num);

		mElements.resize(total_num);
		
		parallel_allocate_for_list<sizeof(ElementType)>(mLists.begin(), mElements.begin(), mElements.size(), mIndex);

		return true;
	}


	template<class ElementType>
	bool ArrayList<ElementType, DeviceType::GPU>::resize(const uint arraySize, const uint eleSize)
	{
		assert(arraySize > 0);
		assert(eleSize > 0);

		if (mIndex.size() != arraySize)
		{
			mIndex.resize(arraySize);
			mLists.resize(arraySize);
		}

		CArray<uint> hIndex;
		hIndex.resize(arraySize);
		int accNum = 0;
		for (size_t i = 0; i < arraySize; i++)
		{
			hIndex[i] = (uint)accNum;
			accNum += eleSize;
		}

		mIndex.assign(hIndex);

		mElements.resize(arraySize*eleSize);

		parallel_allocate_for_list<sizeof(ElementType)>(mLists.begin(), mElements.begin(), mElements.size(), mIndex);

		return true;
	}

	template<typename ElementType>
	template<typename ET2>
	bool ArrayList<ElementType, DeviceType::GPU>::resize(const ArrayList<ET2, DeviceType::GPU>& src) {
		uint arraySize = src.size();
		if (mIndex.size() != arraySize)
		{
			mIndex.resize(arraySize);
			mLists.resize(arraySize);
		}

		mIndex.assign(src.index());
		mElements.resize(src.elementSize());

		parallel_allocate_for_list<sizeof(ElementType)>(mLists.begin(), mElements.begin(), mElements.size(), mIndex);

		return true;
	}

	template<class ElementType>
	void ArrayList<ElementType, DeviceType::GPU>::assign(const ArrayList<ElementType, DeviceType::GPU>& src)
	{
		mIndex.assign(src.index());
		mElements.assign(src.elements());

		mLists.assign(src.lists());

		//redirect the element address
		parallel_init_for_list<sizeof(ElementType)>(mLists.begin(), mElements.begin(), mElements.size(), mIndex);
	}

	template<class ElementType>
	void ArrayList<ElementType, DeviceType::GPU>::assign(const ArrayList<ElementType, DeviceType::CPU>& src)
	{
		mIndex.assign(src.index());
		mElements.assign(src.elements());

		mLists.assign(src.lists());

		//redirect the element address
		parallel_init_for_list<sizeof(ElementType)>(mLists.begin(), mElements.begin(), mElements.size(), mIndex);
	}

	template<class ElementType>
	void ArrayList<ElementType, DeviceType::GPU>::assign(const std::vector<std::vector<ElementType>>& src)
	{
		size_t indNum = src.size();
		CArray<uint> hIndex(indNum);

		CArray<ElementType> hElements;

		size_t eleNum = 0;
		for (int i = 0; i < src.size(); i++)
		{
			hIndex[i] = (uint)eleNum;
			eleNum += src[i].size();

			for (int j = 0; j < src[i].size(); j++)
			{
				hElements.pushBack(src[i][j]);
			}
		}

		CArray<List<ElementType>> lists;
		lists.resize(indNum);
			
		mIndex.assign(hIndex);
		mElements.assign(hElements);
		ElementType* stAdr = mElements.begin();

		eleNum = 0;
		for (int i = 0; i < src.size(); i++)
		{
			size_t num_i = src[i].size();
			List<ElementType> lst;
			lst.assign(stAdr + eleNum, num_i, num_i);
			lists[i] = lst;

			eleNum += src[i].size();
		}

		mLists.assign(lists);
	}

	template<class ElementType>
	bool ArrayList<ElementType, DeviceType::CPU>::resize(uint num)
	{
		assert(num > 0);

		mIndex.resize(num);
		mLists.resize(num);
		
		return true;
	}

	template<class ElementType>
	bool ArrayList<ElementType, DeviceType::GPU>::resize(uint num)
	{
		assert(num > 0);

		mIndex.resize(num);
		mLists.resize(num);

		return true;
	}
}