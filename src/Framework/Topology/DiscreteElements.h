#pragma once
#include "Module/TopologyModule.h"
#include "Primitive3D.h"

namespace dyno
{
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

		DiscreteElements();
		~DiscreteElements() override;

		void scale(Real s);

		DArray<Box3D>& getBoxes() { return m_boxes; }
		DArray<Sphere3D>& getSpheres() { return m_spheres; }

		Box3D getHostBoxes(int i) { return m_hostBoxes[i]; }
		Sphere3D getHostSpheres(int i) { return m_hostSpheres[i]; }

		bool initializeImpl() override;

		void addBox(Box3D box) { m_hostBoxes.push_back(box); }
		void addSphere(Sphere3D sphere) { m_hostSpheres.push_back(sphere); }

	protected:

		DArray<Sphere3D> m_spheres;
		DArray<Box3D> m_boxes;

		
		std::vector<Sphere3D> m_hostSpheres;
		std::vector<Box3D> m_hostBoxes;
	};
}

