#pragma once
#include "Framework/NumericalModel.h"
#include "Framework/FieldVar.h"
#include "Framework/FieldArray.h"
#include "DensityPBDMesh.h"
#include "Attribute.h"
#include "Framework/ModuleTopology.h"
#include "MeshCollision.h"

namespace dyno
{
	template<typename TDataType> class PointSetToPointSet;
	template<typename TDataType> class ParticleIntegrator;
	template<typename TDataType> class NeighborQuery;
	template<typename TDataType> class NeighborTriangleQuery;
	template<typename TDataType> class DensityPBD;
	template<typename TDataType> class SurfaceTension;
	template<typename TDataType> class ImplicitViscosity;
	template<typename TDataType> class Helmholtz;
	template<typename TDataType> class MeshCollision;


	template<typename> class PointSetToPointSet;
	typedef typename TopologyModule::Triangle Triangle;

	template <typename TDataType> class PointSet;


	class ForceModule;
	class ConstraintModule;
	class Attribute;
	/*!
	*	\class	ParticleSystem
	*	\brief	Position-based fluids.
	*
	*	This class implements a position-based fluid solver.
	*	Refer to Macklin and Muller's "Position Based Fluids" for details
	*
	*/
	template<typename TDataType>
	class PositionBasedFluidModelMesh : public NumericalModel
	{
		DECLARE_CLASS_1(PositionBasedFluidModelMesh, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		PositionBasedFluidModelMesh();
		virtual ~PositionBasedFluidModelMesh();

		void step(Real dt) override;

		void setSmoothingLength(Real len) { m_smoothingLength.setValue(len); }
		void setRestDensity(Real rho) { m_restRho = rho; }

		void setIncompressibilitySolver(std::shared_ptr<ConstraintModule> solver);
		void setViscositySolver(std::shared_ptr<ConstraintModule> solver);
		void setSurfaceTensionSolver(std::shared_ptr<ConstraintModule> solver);

		//bool initGhostBoundary();


		DeviceArrayField<Real>* getDensityField()
		{
			return &(m_pbdModule2->m_density);
			//return m_forceDensity;
		}

	public:
		VarField<Real> m_smoothingLength;
		
		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Coord> m_velocity;

		DeviceArrayField<Coord> m_position_all;
		DeviceArrayField<Coord> m_position_ghost;
		DeviceArrayField<Coord> m_velocity_all;
	
		DeviceArrayField<Real> m_massArray;
		DeviceArrayField<Real> PressureFluid;
		DeviceArrayField<Real> m_vn;
		DeviceArrayField<Coord> m_TensionForce;
		DeviceArrayField<Coord> m_forceDensity;
		DeviceArrayField<int> ParticleId;

		DeviceArrayField<Attribute> m_attribute;
		DeviceArrayField<Coord> m_normal;
		DeviceArrayField<int> m_flip;
		
		VarField<int> Start;

		std::shared_ptr<PointSet<TDataType>> m_pSetGhost;

		DeviceArrayField<Coord> TriPoint;
		DeviceArrayField<Coord> TriPointOld;
		DeviceArrayField<Triangle> Tri;

		DeviceArrayField<Real> massTri;

	protected:
		bool initializeImpl() override;

	private:
		int m_pNum;
		Real m_restRho;
		int first = 1;

		//std::shared_ptr<ConstraintModule> m_surfaceTensionSolver;
		std::shared_ptr<ConstraintModule> m_viscositySolver;
		
		std::shared_ptr<ConstraintModule> m_incompressibilitySolver;

		std::shared_ptr<MeshCollision<TDataType>> m_meshCollision;


		std::shared_ptr<DensityPBDMesh<TDataType>> m_pbdModule2;

		std::shared_ptr<ImplicitViscosity<TDataType>> m_visModule;
		std::shared_ptr<SurfaceTension<TDataType>>  m_surfaceTensionSolver;
		std::shared_ptr<Helmholtz<TDataType>> m_Helmholtz;
		std::shared_ptr<PointSetToPointSet<TDataType>> m_mapping;
		std::shared_ptr<ParticleIntegrator<TDataType>> m_integrator;
		std::shared_ptr<NeighborQuery<TDataType>>m_nbrQueryPoint;
		std::shared_ptr<NeighborQuery<TDataType>>m_nbrQueryPointAll;
		std::shared_ptr<NeighborTriangleQuery<TDataType>>m_nbrQueryTri;
	};

#ifdef PRECISION_FLOAT
	template class PositionBasedFluidModelMesh<DataType3f>;
#else
	template class PositionBasedFluidModelMesh<DataType3d>;
#endif
}