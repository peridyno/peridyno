#include "DiscreteElements.h"

namespace dyno
{
	IMPLEMENT_TCLASS(DiscreteElements, TDataType)

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
		return tetIndex() + this->getTets().size();
	}

	template<typename TDataType>
	uint DiscreteElements<TDataType>::boxIndex()
	{
		return sphereIndex() + this->getSpheres().size();
	}

	template<typename TDataType>
	ElementOffset DiscreteElements<TDataType>::calculateElementOffset()
	{
		ElementOffset elementOffset;
		elementOffset.setSphereRange(sphereIndex(), sphereIndex() + this->getSpheres().size());
		elementOffset.setBoxRange(boxIndex(), boxIndex() + this->getBoxes().size());
		elementOffset.setTetRange(tetIndex(), tetIndex() + this->getTets().size());
		elementOffset.setCapsuleRange(capsuleIndex(), capsuleIndex() + this->getCaps().size());
		elementOffset.setTriangleRange(triangleIndex(), triangleIndex() + this->getTris().size());

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

	DEFINE_CLASS(DiscreteElements);
}