#include "DiscreteElements.h"

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
// 		m_hostBoxes.clear();
// 		m_hostSpheres.clear();
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::scale(Real s)
	{
	}

	template<typename TDataType>
	uint DiscreteElements<TDataType>::totalSize()
	{
		return m_boxes.size() + m_spheres.size() + m_tets.size() + m_caps.size() + m_tris.size();
	}

	template<typename TDataType>
	uint DiscreteElements<TDataType>::sphereIndex()
	{
		return 0;
	}

	template<typename TDataType>
	uint DiscreteElements<TDataType>::triangleIndex()
	{
		return capsuleIndex() + this->getCaps().size();
	}

	template<typename TDataType>
	uint DiscreteElements<TDataType>::tetIndex()
	{
		return boxIndex() + this->getBoxes().size();
	}

	template<typename TDataType>
	uint DiscreteElements<TDataType>::capsuleIndex()
	{
		return tetIndex() + this->getCaps().size();
	}

	template<typename TDataType>
	uint DiscreteElements<TDataType>::boxIndex()
	{
		return sphereIndex() + this->getSpheres().size();
	}

// 	template<typename TDataType>
// 	bool DiscreteElements<TDataType>::initializeImpl()
// 	{
// 		m_spheres.resize(m_hostSpheres.size());
// 		m_boxes.resize(m_hostBoxes.size());
// 		m_tets.resize(m_hostTets.size());
// 		m_caps.resize(m_hostCaps.size());
// 
// 		if (m_spheres.size() > 0)
// 			m_spheres.assign(m_hostSpheres);
// 		if(m_boxes.size() > 0)
// 			m_boxes.assign(m_hostBoxes);
// 		if (m_tets.size() > 0)
// 			m_tets.assign(m_hostTets);
// 		if (m_caps.size() > 0)
// 			m_caps.assign(m_hostCaps);
// 
// 
// 		//printf("%d\n", m_boxes.size());
// 
// 		return true;
// 	}

	template<typename TDataType>
	ElementOffset DiscreteElements<TDataType>::calculateElementOffset()
	{
		ElementOffset elementOffset;
		elementOffset.sphereStart = sphereIndex();
		elementOffset.sphereEnd = sphereIndex() + this->getSpheres().size();
		elementOffset.boxOffset = boxIndex();
		elementOffset.boxEnd = boxIndex() + this->getBoxes().size();
		elementOffset.tetOffset = tetIndex();
		elementOffset.tetEnd = tetIndex() + this->getTets().size();
		elementOffset.segOffset = capsuleIndex();
		elementOffset.segEnd = capsuleIndex() + this->getCaps().size();
		elementOffset.triOffset = triangleIndex();
		elementOffset.triEnd = triangleIndex() + this->getTris().size();

		return elementOffset;
	}


	template<typename TDataType>
	void DiscreteElements<TDataType>::setBoxes(DArray<Box3D>& boxes)
	{
		m_boxes.assign(boxes);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setSpheres(DArray<Sphere3D>& spheres)
	{
		m_spheres.assign(spheres);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setTetSDF(DArray<Real>& sdf)
	{
		m_tet_sdf.assign(sdf);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setTets(DArray<Tet3D>& tets)
	{
		m_tets.assign(tets);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setCapsules(DArray<Capsule3D>& capsules)
	{
		m_caps.assign(capsules);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setTetBodyId(DArray<int>& body_id)
	{
		m_tet_body_mapping.assign(body_id);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setTetElementId(DArray<TopologyModule::Tetrahedron>& element_id)
	{
		m_tet_element_id.assign(element_id);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setTriangles(DArray<Triangle3D>& triangles)
	{
		m_tris.assign(triangles);
	}

#ifdef PRECISION_FLOAT
	template class DiscreteElements<DataType3f>;
#else
	template class DiscreteElements<DataType3d>;
#endif
}