#pragma once
#include <set>
#include "Array/Array.h"

namespace dyno
{
	template<typename ElementType>
	class NeighborList
	{
	public:
		NeighborList()
			: m_maxNum(0)
		{
		};

		NeighborList(int n, int maxNbr)
			: m_maxNum(maxNbr)
		{
			resize(n, maxNbr);
		};

		~NeighborList() {};

		DYN_FUNC uint size() { return m_index.size(); }
		DYN_FUNC uint getElementSize() { return m_elements.size(); }

		GPU_FUNC int getNeighborSize(int i)
		{ 
			if (!isLimited())
			{
				if (i >= m_index.size() - 1)
				{
					return m_elements.size() - m_index[i];
				}
				return m_index[i + 1] - m_index[i];
			}
			else
			{
				return m_index[i];
			}
		}

		DYN_FUNC int getNeighborLimit()
		{
			return m_maxNum;
		}

		GPU_FUNC void setNeighborSize(int i, int num)
		{
			if (isLimited())
				m_index[i] = num;
		}

		GPU_FUNC ElementType getElement(int i, int j) {
			if (!isLimited())
				return m_elements[m_index[i] + j];
			else
				return m_elements[m_maxNum*i + j];
		};
		
		GPU_FUNC int getElementIndex(int i, int j) {
			if (!isLimited())
				return (m_index[i] + j);
			else
				return (m_maxNum * i + j);
		};

		GPU_FUNC void setElement(int i, int j, ElementType elem) {
			if (!isLimited())
				m_elements[m_index[i] + j] = elem;
			else
				m_elements[m_maxNum*i + j] = elem;
		}

		DYN_FUNC bool isLimited()
		{
			return m_maxNum > 0;
		}

		void resize(int n, int maxNbr = 0) {
			m_index.resize(n);
			if (maxNbr != 0)
			{
				setNeighborLimit(maxNbr);
			}
			else
			{
				setDynamic();
			}
		}
		
		void release()
		{
			m_elements.clear();
			m_index.clear();
		}

		void setNeighborLimit(int nbrMax)
		{
			m_maxNum = nbrMax;
			m_elements.resize(m_maxNum*m_index.size());
		}

		void setDynamic()
		{
			m_maxNum = 0;
		}

		void copyFrom(NeighborList<ElementType>& neighborlist)
		{
			m_maxNum = neighborlist.m_maxNum;
			if (m_elements.size() != neighborlist.m_elements.size())
				m_elements.resize(neighborlist.m_elements.size());

			m_elements.assign(neighborlist.m_elements);

			if (m_index.size() != neighborlist.m_index.size())
				m_index.resize(neighborlist.m_index.size());

			m_index.assign(neighborlist.m_index);
			
		}

		void copyFrom(int maxNum, std::vector<ElementType>& elements, std::vector<int> index)
		{
			m_maxNum = maxNum;
			if (m_elements.size() != elements.size())
				m_elements.resize(elements.size());

			m_elements.assign(elements);

			if (m_index.size() != index.size())
				m_index.resize(index.size());

			m_index.assign(index);
		}

		DArray<int>& getIndex() { return m_index; }
		DArray<ElementType>& getElements() { return m_elements; }

	private:

		int m_maxNum;
		DArray<ElementType> m_elements;
		DArray<int> m_index;
	};
}