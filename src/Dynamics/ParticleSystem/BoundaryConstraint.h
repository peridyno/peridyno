#pragma once
#include "Framework/ModuleConstraint.h"
#include "Framework/FieldArray.h"

namespace dyno {

	template<typename TDataType> class DistanceField3D;

	template<typename TDataType>
	class BoundaryConstraint : public ConstraintModule
	{
		DECLARE_CLASS_1(BoundaryConstraint, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		BoundaryConstraint();
		~BoundaryConstraint() override;

		bool constrain() override;

		bool constrain(GArray<Coord>& position, GArray<Coord>& velocity, Real dt);

		void load(std::string filename, bool inverted = false);
		void setCube(Coord lo, Coord hi, Real distance, bool inverted = false);
		void setSphere(Coord center, Real r, Real distance, bool inverted = false);

	public:
		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Coord> m_velocity;

		DEF_VAR(TangentialFriction, Real, Real(0.95), "Tangential friction");
		DEF_VAR(NormalFriction, Real, Real(0), "Normal friction");

		std::shared_ptr<DistanceField3D<TDataType>> m_cSDF;
	};

#ifdef PRECISION_FLOAT
template class BoundaryConstraint<DataType3f>;
#else
template class BoundaryConstraint<DataType3d>;
#endif

}
