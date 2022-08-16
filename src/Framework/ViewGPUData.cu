#pragma once
#include "ViewGPUData.h"

namespace dyno
{
    template<typename TDataType>
    ViewGPUData<TDataType>::ViewGPUData()
    {
        for(int i = 0; i < LIMIT_INDEX; ++ i)
            m_view[i] = EMat::Zero();
        m_size = 0;
    }
    
    template<typename TDataType>
    ViewGPUData<TDataType>::ViewGPUData(int size)
    {
        for(int i = 0; i < LIMIT_INDEX; ++ i)
            m_view[i] = EMat::Zero();
        this->resize(size);
    }

    template<typename TDataType>
    ViewGPUData<TDataType>::~ViewGPUData()
    {
        m_GPU.clear();
        m_CPU.clear();
    }

    template<typename TDataType>
    void ViewGPUData<TDataType>::resize(int size)
    {
        if (size > LIMIT_SIZE) size = LIMIT_SIZE;
        if (size < m_size)
            for (int i = 0; i < LIMIT_INDEX; ++i)
                m_view[i] = EMat::Zero();
        m_GPU.resize(size);
        m_CPU.resize(size);
        m_size = size;
    }

    template<typename TDataType>
    bool ViewGPUData<TDataType>::view(int index)
    {
        if (m_size != m_GPU.size()) return false;

        m_CPU.assign(m_GPU);
        for (int i = 0; i < m_size; ++i)
        {
            m_view[index](i % LINE_SIZE, i / LINE_SIZE) = m_CPU[i];
        }

        return true;
    }

    template <typename Real>
	__global__ void VGPUReal(
        DArray<Real> p,
        DArray<Real> m,
        int size)
    {
        int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if(pId >= size) return;
        m[pId] = p[pId];
    }

    template<typename TDataType>
    bool ViewGPUData<TDataType>::viewIm(DArray<Real> &p_GPU)
    {
        cuExecute(m_size,
                VGPUReal,
                p_GPU,
                m_GPU,
                m_size);
            cuSynchronize();
        status = this->view(0);
        return status;
    }    

    template <typename Coord, typename Real>
	__global__ void VGPUCoord(
        DArray<Coord> p,
        DArray<Real> m,
        int index,
        int size)
    {
        int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if(pId >= size) return;    
        m[pId] = p[pId][index];
    }

    template<typename TDataType>
    bool ViewGPUData<TDataType>::viewIm(DArray<Coord> &p_GPU)
    {
        status = true;
        for (int i = 0; i < 3; ++i)
        {
            cuExecute(m_size,
                VGPUCoord,
                p_GPU,
                m_GPU,
                i,
                m_size);
            cuSynchronize();
            if (!this->view(i)) status = false;
        }
        return status;
    }   

    

#ifdef PRECISION_FLOAT
	template class ViewGPUData<DataType3f>;

#else
    template class ViewGPUData<DataType3d>;
#endif    
}