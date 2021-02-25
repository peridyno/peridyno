#pragma once
#include "Framework/NumericalModel.h"
#include "Framework/FieldVar.h"
#include "Framework/FieldArray.h"

namespace dyno
{
	/*!
	*	\class	ParticleSystem
	*	\brief	Position-based fluids.
	*
	*	This class implements a position-based fluid solver.
	*	Refer to Macklin and Muller's "Position Based Fluids" for details
	*
	*/
	template<typename TDataType>
	class ShallowWaterEquationModel : public NumericalModel
	{
		DECLARE_CLASS_1(ShallowWaterEquationModel, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ShallowWaterEquationModel();
		virtual ~ShallowWaterEquationModel();

		void step(Real dt) override;

		void setDistance(Real distance) { this->distance = distance; }
		void setRelax(Real relax) { this->relax = relax; }
	public:
		
		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Coord> m_velocity;
		DeviceArrayField<Coord> m_accel;
		
		DeviceArrayField<Coord> solid;
		DeviceArrayField<Coord> normal;
		DeviceArrayField<int>  isBound;
		DeviceArrayField<Real> h;//solid_pos + h*Normal = m_position
		
		NeighborField<int>	neighborIndex;

	protected:
		bool initializeImpl() override;

	private:
		int m_pNum;
		Real distance;
		Real relax;
	};

#ifdef PRECISION_FLOAT
	template class ShallowWaterEquationModel<DataType3f>;
#else
	template class ShallowWaterEquationModel<DataType3d>;
#endif
}