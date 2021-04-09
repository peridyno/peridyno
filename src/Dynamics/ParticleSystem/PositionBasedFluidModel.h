#pragma once
#include "Framework/NumericalModel.h"
#include "DensityPBD.h"

namespace dyno
{
	template<typename TDataType> class PointSetToPointSet;
	template<typename TDataType> class ParticleIntegrator;
	template<typename TDataType> class NeighborPointQuery;
	template<typename TDataType> class DensityPBD;
	template<typename TDataType> class ImplicitViscosity;
	class ForceModule;
	class ConstraintModule;
	/*!
	*	\class	ParticleSystem
	*	\brief	Position-based fluids.
	*
	*	This class implements a position-based fluid solver.
	*	Refer to Macklin and Muller's "Position Based Fluids" for details
	*
	*/
	template<typename TDataType>
	class PositionBasedFluidModel : public NumericalModel
	{
		DECLARE_CLASS_1(PositionBasedFluidModel, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		PositionBasedFluidModel();
		virtual ~PositionBasedFluidModel();

		void step(Real dt) override;

		void setSmoothingLength(Real len) { m_smoothingLength.setValue(len); }
		void setRestDensity(Real rho) { m_restRho = rho; }

		void setIncompressibilitySolver(std::shared_ptr<ConstraintModule> solver);
		void setViscositySolver(std::shared_ptr<ConstraintModule> solver);
		void setSurfaceTensionSolver(std::shared_ptr<ForceModule> solver);

		DeviceArrayField<Real>* getDensityField()
		{
			return m_pbdModule->outDensity();
		}

	public:
		VarField<Real> m_smoothingLength;

		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Coord> m_velocity;
		DeviceArrayField<Coord> m_forceDensity;

	protected:
		bool initializeImpl() override;

	private:
		int m_pNum;
		Real m_restRho;

		std::shared_ptr<ForceModule> m_surfaceTensionSolver;
		std::shared_ptr<ConstraintModule> m_viscositySolver;
		std::shared_ptr<ConstraintModule> m_incompressibilitySolver;

		std::shared_ptr<DensityPBD<TDataType>> m_pbdModule;
		std::shared_ptr<ImplicitViscosity<TDataType>> m_visModule;

		std::shared_ptr<PointSetToPointSet<TDataType>> m_mapping;
		std::shared_ptr<ParticleIntegrator<TDataType>> m_integrator;
		std::shared_ptr<NeighborPointQuery<TDataType>>m_nbrQuery;
	};
}