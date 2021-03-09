#include "DiscreteElements.h"
#include <fstream>
#include <iostream>
#include <sstream>

namespace dyno
{
	IMPLEMENT_CLASS_1(DiscreteElements, TDataType)

	template<typename TDataType>
	DiscreteElements<TDataType>::DiscreteElements()
		: TopologyModule()
	{
	}

	template<typename TDataType>
	DiscreteElements<TDataType>::~DiscreteElements()
	{
		m_hostBoxes.clear();
		m_hostSpheres.clear();
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::scale(Real s)
	{
	}

	template<typename TDataType>
	bool DiscreteElements<TDataType>::initializeImpl()
	{
		m_spheres.resize(m_hostSpheres.size());
		m_boxes.resize(m_hostBoxes.size());

		if(m_spheres.size() > 0)
			m_spheres.assign(m_hostSpheres);
		if(m_boxes.size() > 0)
			m_boxes.assign(m_hostBoxes);

		return true;
	}
}