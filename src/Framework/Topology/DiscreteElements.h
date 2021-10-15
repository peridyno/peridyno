#pragma once
#include "Module/TopologyModule.h"
#include "Primitive3D.h"

namespace dyno
{
	enum ElementType
	{
		ET_BOX = 1,
		ET_TET = 2,
		ET_CAPSULE = 4,
		ET_SPHERE = 8,
		ET_TRI = 16,
		ET_Other = 0x80000000
	};

	struct ElementOffset
	{
		int sphereStart;
		int sphereEnd;
		int boxOffset;
		int boxEnd;
		int tetOffset;
		int tetEnd;
		int segOffset;
		int segEnd;
		int triOffset;
		int triEnd;
	};

	DYN_FUNC inline ElementType checkElementType(int id, ElementOffset offset)
	{
		if (id >= offset.sphereStart && id < offset.sphereEnd)
			return ET_SPHERE;

		if (id >= offset.boxOffset && id < offset.boxEnd)
			return ET_BOX;

		if (id >= offset.tetOffset && id < offset.tetEnd)
			return ET_TET;

		if (id >= offset.segOffset && id < offset.segEnd)
			return ET_CAPSULE;

		if (id >= offset.triOffset && id < offset.triEnd)
			return ET_TRI;
	}

	template<typename TDataType>
	class DiscreteElements : public TopologyModule
	{
		DECLARE_CLASS_1(DiscreteElements, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename TSphere3D<Real> Sphere3D;
		typedef typename TOrientedBox3D<Real> Box3D;
		typedef typename TTet3D<Real> Tet3D;

		DiscreteElements();
		~DiscreteElements() override;

		void scale(Real s);

		uint totalSize();

		uint sphereIndex();
		uint boxIndex();
		uint capsuleIndex();
		uint tetIndex();
		uint triangleIndex();

		ElementOffset calculateElementOffset();

		void setBoxes(DArray<Box3D>& boxes);
		void setSpheres(DArray<Sphere3D>& spheres);
		void setTets(DArray<Tet3D>& tets);
		void setCapsules(DArray<Capsule3D>& capsules);
		void setTriangles(DArray<Triangle3D>& triangles);
		void setTetSDF(DArray<Real>& sdf);

		DArray<Box3D>&		getBoxes() { return m_boxes; }
		DArray<Sphere3D>&	getSpheres() { return m_spheres; }
		DArray<Tet3D>&		getTets() { return m_tets; }
		DArray<Capsule3D>&	getCaps() { return m_caps; }
		DArray<Triangle3D>& getTris() { return m_tris; }

		void setTetBodyId(DArray<int>& body_id);
		void setTetElementId(DArray<TopologyModule::Tetrahedron>& element_id);

		DArray<Real>&		getTetSDF() { return m_tet_sdf; }
		DArray<int>&		getTetBodyMapping() { return m_tet_body_mapping; }
		DArray<TopologyModule::Tetrahedron>& getTetElementMapping() { return m_tet_element_id; }

	protected:
		DArray<Sphere3D> m_spheres;
		DArray<Box3D> m_boxes;
		DArray<Tet3D> m_tets;
		DArray<Capsule3D> m_caps;
		DArray<Triangle3D> m_tris;
		
		DArray<Real> m_tet_sdf;
		DArray<int> m_tet_body_mapping;
		DArray<TopologyModule::Tetrahedron> m_tet_element_id;
	};
}

