#pragma once
#include "Module/TopologyModule.h"
#include "Primitive3D.h"


namespace dyno
{
	enum ElementType
	{
		CT_SPHERE,
		CT_BOX,
		CT_TET,
		CT_SEG,
		CT_TRI
	};

	struct ElementOffset
	{
		int boxOffset;
		int tetOffset;
		int segOffset;
		int triOffset;
	};

	DYN_FUNC inline ElementType checkElementType(int id, ElementOffset offset)
	{
		return id < offset.boxOffset ?
			CT_SPHERE : (
				id < offset.tetOffset ? CT_BOX : (
					id < offset.segOffset ? CT_TET : (
						id < offset.triOffset ? CT_SEG : CT_TRI)));
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

		DArray<Box3D>& getBoxes() { return m_boxes; }
		DArray<Sphere3D>& getSpheres() { return m_spheres; }
		DArray<Tet3D>& getTets() { return m_tets; }
		DArray<Capsule3D>& getCaps() { return m_caps; }
		DArray<Triangle3D>& getTris() { return m_tris; }

		DArray<Real>& getTetSDF() { return m_tet_sdf; }
		DArray<int>& getTetBodyMapping() { return m_tet_body_mapping; }
		DArray<TopologyModule::Tetrahedron>& getTetElementMapping() { return m_tet_element_id; }

		void setBoxes(DArray<Box3D>& boxes);
		void setSpheres(DArray<Sphere3D>& spheres);
		void setTets(DArray<Tet3D>& tets);
		void setCapsules(DArray<Capsule3D>& capsules);
		void setTriangles(DArray<Triangle3D>& triangles);
		void setTetSDF(DArray<Real>& sdf);

		void setTetBodyId(DArray<int>& body_id);
		void setTetElementId(DArray<TopologyModule::Tetrahedron>& element_id);


		Real getSize()
		{
			return m_boxes.size() + m_spheres.size() + m_tets.size() + m_caps.size() + m_tris.size();
		}

		Box3D getHostBoxes(int i) { return m_hostBoxes[i]; }
		Sphere3D getHostSpheres(int i) { return m_hostSpheres[i]; }
		Tet3D getHostTets(int i) { return m_hostTets[i]; }
		Capsule3D getHostCaps(int i) { return m_hostCaps[i]; }

		bool initializeImpl() override;

		void addBox(Box3D box) { m_hostBoxes.push_back(box); }
		void addSphere(Sphere3D sphere) { m_hostSpheres.push_back(sphere); }
		void addTet(Tet3D tet) { m_hostTets.push_back(tet); }
		void addCap(Capsule3D cap) { m_hostCaps.push_back(cap); }

		ElementOffset calculateElementOffset();

	protected:

		DArray<Sphere3D> m_spheres;
		DArray<Box3D> m_boxes;
		DArray<Tet3D> m_tets;
		DArray<Capsule3D> m_caps;
		DArray<Triangle3D> m_tris;
		
		DArray<Real> m_tet_sdf;
		DArray<int> m_tet_body_mapping;
		DArray<TopologyModule::Tetrahedron> m_tet_element_id;

		std::vector<Sphere3D> m_hostSpheres;
		std::vector<Box3D> m_hostBoxes;
		std::vector<Tet3D> m_hostTets;
		std::vector<Capsule3D> m_hostCaps;
	};
}

