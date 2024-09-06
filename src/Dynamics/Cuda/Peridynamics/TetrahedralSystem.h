#pragma once
#include "Node.h"

#include "Topology/TetrahedronSet.h"

namespace dyno
{
	/*!
	*	\class	TetSystem
	*	\brief	This class represents the base class for more advanced particle-based nodes.
	*/
	template<typename TDataType>
	class TetrahedralSystem : public Node
	{
		DECLARE_TCLASS(TetrahedralSystem, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		TetrahedralSystem();
		~TetrahedralSystem() override;

		DEF_ARRAY_STATE(Coord, NormalSDF, DeviceType::GPU, "");
		DEF_VAR(Bool, SDF, false, "has SDF");

		/**
		 * @brief A topology
		 */
		DEF_INSTANCE_STATE(TetrahedronSet<TDataType>, TetrahedronSet, "Topology");

		/**
		 * @brief Vertex position
		 */
		DEF_ARRAY_STATE(Coord, Position, DeviceType::GPU, "Vertex position");

		/**
		 * @brief Vertex velocity
		 */
		DEF_ARRAY_STATE(Coord, Velocity, DeviceType::GPU, "Vertex velocity");

		/**
		 * @brief Vertex velocity
		 */
		DEF_ARRAY_STATE(Coord, Force, DeviceType::GPU, "Vertex force");

	public:
		void loadVertexFromFile(std::string filename);
		void loadVertexFromGmshFile(std::string filename);

		virtual bool translate(Coord t);
		virtual bool scale(Real s);
		virtual bool rotate(Quat<Real> q);

	protected:
		void updateTopology() override;
		void resetStates() override;
	};
}