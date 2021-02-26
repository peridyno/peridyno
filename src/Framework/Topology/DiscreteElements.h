#pragma once
#include "Framework/ModuleTopology.h"
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

		GArray<Box3D>& getBoxes() { return m_boxes; }
		GArray<Sphere3D>& getSpheres() { return m_spheres; }

		Box3D getHostBoxes(int i) { return m_hostBoxes[i]; }
		Sphere3D getHostSpheres(int i) { return m_hostSpheres[i]; }

		bool initializeImpl() override;

		void addBox(Box3D box) { m_hostBoxes.push_back(box); }
		void addSphere(Sphere3D sphere) { m_hostSpheres.push_back(sphere); }

	protected:

		GArray<Sphere3D> m_spheres;
		GArray<Box3D> m_boxes;

		
		std::vector<Sphere3D> m_hostSpheres;
		std::vector<Box3D> m_hostBoxes;
	};


#ifdef PRECISION_FLOAT
	template class DiscreteElements<DataType3f>;
#else
	template class DiscreteElements<DataType3d>;
#endif
}

