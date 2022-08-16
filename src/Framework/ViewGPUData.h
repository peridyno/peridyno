    #pragma once
#include "Vector.h"
#include "Module.h"
#include <Eigen/Core>
#include "Module/TopologyModule.h"

#define LIMIT_SIZE 10000
#define LINE_SIZE 100
#define ROW_SIZE 100
#define LIMIT_INDEX 4

namespace dyno
{
	/*!
	*	\class	ViewGPUData
	*	\brief	View GPU Data with Image Watch
	*/    
	template<typename TDataType>
	class ViewGPUData
	{
	public:

		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
        typedef typename Eigen::Matrix<Real, LINE_SIZE, ROW_SIZE> EMat;
        
		typedef VectorND<Real, 3> Pair3f;

		ViewGPUData();
		ViewGPUData(int size);
		~ViewGPUData();

        DArray<Real>& getGPU() {return m_GPU;}
        CArray<Real>& getCPU() {return m_CPU;}
        
        void resize(int size);
		
		bool viewIm(DArray<Real> &p_GPU);
		bool viewIm(DArray<Coord> &p_GPU);
		bool viewIm(DArray<Pair3f> &p_GPU);

        bool view(int index);

	private:
		DArray<Real> m_GPU;
        CArray<Real> m_CPU;
        int m_size;
        EMat m_view[LIMIT_INDEX];
		bool status = true;
	};
}