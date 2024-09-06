#pragma once
#include "Module/ConstraintModule.h"

namespace dyno {

	template<typename TDataType>
	class FixedPoints : public ConstraintModule
	{
		DECLARE_TCLASS(FixedPoints, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		FixedPoints();
		~FixedPoints() override;

		void addFixedPoint(int id, Coord pt);
		void removeFixedPoint(int id);

		void clear();

	protected:
		
		void constrain() override;

		void constrainPositionToPlane(Coord pos, Coord dir);

	public:
		/**
		* @brief Particle position
		*/
		//DeviceArrayField<Coord> m_position;

		/**
		* @brief Particle velocity
		*/
		//DeviceArrayField<Coord> m_velocity;

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "");

		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "");

		DeviceArrayField<int> FixedIds;
		DeviceArrayField<Coord> FixedPos;

	protected:
		virtual bool initializeImpl() override;

		FieldID m_initPosID;

	private:
		void updateContext();

		bool bUpdateRequired = false;

		std::map<int, Coord> m_fixedPts;

		std::vector<int> m_bFixed_host;
		std::vector<Coord> m_fixed_positions_host;

		DArray<int> m_bFixed;
		DArray<Coord> m_fixed_positions;
	};
}
