#include "GridSet.h"


namespace dyno 
{
	IMPLEMENT_TCLASS(GridSet, TDataType)

	template<typename TDataType>
	GridSet<TDataType>::GridSet()
	{
		m_dx = 1.0;
		m_origin = Coord(0.0, 0.0, 0.0);
		m_ni = 0;
		m_nj = 0;
		m_nk = 0;
	}

	template<typename TDataType>
	GridSet<TDataType>::~GridSet()
	{
	}

	template<typename TDataType>
	void GridSet<TDataType>::setUniGrid(int ni, int nj, int nk, Real dxmm, Coord lo_center)
	{
		m_ni = ni;
		m_nj = nj;
		m_nk = nk;
		m_dx = dxmm;
		m_origin = lo_center;
	}
	
	template<typename TDataType>
	void GridSet<TDataType>::setNijk(int ni, int nj, int nk)
	{
		m_ni = ni;
		m_nj = nj;
		m_nk = nk;
	}

	DEFINE_CLASS(GridSet);
}
