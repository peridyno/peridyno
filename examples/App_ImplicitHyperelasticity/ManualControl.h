#pragma once
#include "Framework/ModuleCustom.h"
#include "ParticleSystem/Attribute.h"

namespace dyno {

	template<typename TDataType>
	class ManualControl : public CustomModule
	{
		DECLARE_CLASS_1(ManualControl, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ManualControl();
		~ManualControl() override;

		void begin() override;
		void applyCustomBehavior() override;


		void addFixedPoint(int id, Coord pt);
		void removeFixedPoint(int id);

		void clear();

	public:
		/**
		* @brief Particle position
		*/
		DEF_EMPTY_IN_ARRAY(Position, Coord, DeviceType::GPU, "");

		/**
		* @brief Particle velocity
		*/
		DEF_EMPTY_IN_ARRAY(Velocity, Coord, DeviceType::GPU, "");

		DEF_EMPTY_IN_ARRAY(Attribute, Attribute, DeviceType::GPU, "");

	private:
		void updateContext();

		bool bUpdateRequired = false;

		std::map<int, Coord> m_fixedPts;

		std::vector<int> m_bFixed_host;
		std::vector<Coord> m_fixed_positions_host;

		GArray<int> m_bFixed;
		GArray<Coord> m_fixed_positions;
	};

#ifdef PRECISION_FLOAT
template class ManualControl<DataType3f>;
#else
template class ManualControl<DataType3d>;
#endif

}
