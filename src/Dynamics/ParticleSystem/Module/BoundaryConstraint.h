#pragma once
#include "Module/ConstraintModule.h"

namespace dyno {

	template<typename TDataType> class DistanceField3D;

	template<typename TDataType>
	class BoundaryConstraint : public ConstraintModule
	{
		//DECLARE_TCLASS(BoundaryConstraint, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		BoundaryConstraint();
		~BoundaryConstraint() override;

		void constrain() override;

		bool constrain(DArray<Coord>& position, DArray<Coord>& velocity, Real dt);

		void load(std::string filename, bool inverted = false);
		void setCube(Coord lo, Coord hi, Real distance, bool inverted = false);
		void setSphere(Coord center, Real r, Real distance, bool inverted = false);

	public:
		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Coord> m_velocity;

		DEF_VAR(Real, TangentialFriction, Real(0.95), "Tangential friction");
		DEF_VAR(Real, NormalFriction, Real(0), "Normal friction");

		std::shared_ptr<DistanceField3D<TDataType>> m_cSDF;
	};

	//IMPLEMENT_TCLASS(BoundaryConstraint, TDataType)
}
