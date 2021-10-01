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
		m_tets.resize(m_hostTets.size());
		m_caps.resize(m_hostCaps.size());

		if (m_spheres.size() > 0)
			m_spheres.assign(m_hostSpheres);
		if(m_boxes.size() > 0)
			m_boxes.assign(m_hostBoxes);
		if (m_tets.size() > 0)
			m_tets.assign(m_hostTets);
		if (m_caps.size() > 0)
			m_caps.assign(m_hostCaps);


		//printf("%d\n", m_boxes.size());

		return true;
	}

	template<typename TDataType>
	ElementOffset DiscreteElements<TDataType>::calculateElementOffset()
	{
		ElementOffset elementOffset;
		elementOffset.boxOffset = this->getSpheres().size();
		elementOffset.tetOffset = this->getSpheres().size() + this->getBoxes().size();
		elementOffset.segOffset = this->getSpheres().size() + this->getBoxes().size() + this->getTets().size();
		elementOffset.triOffset = this->getSpheres().size() + this->getBoxes().size() + this->getTets().size() + this->getCaps().size();

		return elementOffset;
	}


	template<typename TDataType>
	void DiscreteElements<TDataType>::setBoxes(DArray<Box3D>& boxes)
	{
		m_boxes.resize(boxes.size());
		cudaMemcpy(m_boxes.begin(), boxes.begin(), m_boxes.size() * sizeof(Box3D), cudaMemcpyDeviceToDevice);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setSpheres(DArray<Sphere3D>& spheres)
	{
		m_spheres.resize(spheres.size());
		cudaMemcpy(m_spheres.begin(), spheres.begin(), m_spheres.size() * sizeof(Sphere3D), cudaMemcpyDeviceToDevice);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setTetSDF(DArray<Real>& sdf)
	{
		m_tet_sdf.resize(sdf.size());
		cudaMemcpy(m_tet_sdf.begin(), sdf.begin(), m_tet_sdf.size() * sizeof(Real), cudaMemcpyDeviceToDevice);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setTets(DArray<Tet3D>& tets)
	{
		m_tets.resize(tets.size());
		cudaMemcpy(m_tets.begin(), tets.begin(), m_tets.size() * sizeof(Tet3D), cudaMemcpyDeviceToDevice);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setCapsules(DArray<Capsule3D>& capsules)
	{
		m_caps.resize(capsules.size());
		cudaMemcpy(m_caps.begin(), capsules.begin(), m_caps.size() * sizeof(Capsule3D), cudaMemcpyDeviceToDevice);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setTetBodyId(DArray<int>& body_id)
	{
		m_tet_body_mapping.resize(body_id.size());
		cudaMemcpy(m_tet_body_mapping.begin(), body_id.begin(), m_tet_body_mapping.size() * sizeof(int), cudaMemcpyDeviceToDevice);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setTetElementId(DArray<TopologyModule::Tetrahedron>& element_id)
	{
		m_tet_element_id.resize(element_id.size());
		cudaMemcpy(m_tet_element_id.begin(), element_id.begin(), element_id.size() * sizeof(TopologyModule::Tetrahedron), cudaMemcpyDeviceToDevice);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setTriangles(DArray<Triangle3D>& triangles)
	{
	
		m_tris.resize(triangles.size());
		cudaMemcpy(m_tris.begin(), triangles.begin(), triangles.size() * sizeof(Triangle3D), cudaMemcpyDeviceToDevice);
	}
}