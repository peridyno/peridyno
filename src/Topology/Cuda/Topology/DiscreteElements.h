#pragma once
#include "Module/TopologyModule.h"
#include "Primitive/Primitive3D.h"

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
	public:
		DYN_FUNC inline uint sphereIndex() { return sphereStart; }
		DYN_FUNC inline uint boxIndex() { return boxStart; }
		DYN_FUNC inline uint tetIndex() { return tetStart; }
		DYN_FUNC inline uint capsuleIndex() { return capStart; }
		DYN_FUNC inline uint triangleIndex() { return triStart; }

		DYN_FUNC inline void setSphereRange(uint startIndex, uint endIndex) { 
			sphereStart = startIndex;
			sphereEnd = endIndex;
		}

		DYN_FUNC inline void setBoxRange(uint startIndex, uint endIndex) {
			boxStart = startIndex;
			boxEnd = endIndex;
		}

		DYN_FUNC inline void setTetRange(uint startIndex, uint endIndex) {
			tetStart = startIndex;
			tetEnd = endIndex;
		}

		DYN_FUNC inline void setCapsuleRange(uint startIndex, uint endIndex) {
			capStart = startIndex;
			capEnd = endIndex;
		}

		DYN_FUNC inline void setTriangleRange(uint startIndex, uint endIndex) {
			triStart = startIndex;
			triEnd = endIndex;
		}

		DYN_FUNC inline ElementType checkElementType(uint id)
		{
			if (id >= sphereStart && id < sphereEnd)
				return ET_SPHERE;

			if (id >= boxStart && id < boxEnd)
				return ET_BOX;

			if (id >= tetStart && id < tetEnd)
				return ET_TET;

			if (id >= capStart && id < capEnd)
				return ET_CAPSULE;

			if (id >= triStart && id < triEnd)
				return ET_TRI;
		}

	private:
		uint sphereStart;
		uint sphereEnd;
		uint boxStart;
		uint boxEnd;
		uint tetStart;
		uint tetEnd;
		uint capStart;
		uint capEnd;
		uint triStart;
		uint triEnd;
	};

	template<typename TDataType>
	class DiscreteElements : public TopologyModule
	{
		DECLARE_TCLASS(DiscreteElements, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename ::dyno::TSphere3D<Real> Sphere3D;
		typedef typename ::dyno::TOrientedBox3D<Real> Box3D;
		typedef typename ::dyno::TTet3D<Real> Tet3D;

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

