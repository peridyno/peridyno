#pragma once
#include "Node.h"

#include "Topology/TriangleSet.h"

namespace dyno
{
	/*!
	*	\class	TetSystem
	*	\brief	This class represents the base class for more advanced particle-based nodes.
	*/
	template<typename TDataType>
	class TriangularSystem : public Node
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		TriangularSystem();
		~TriangularSystem() override;

		virtual bool translate(Coord t);
		virtual bool scale(Real s);

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");

		/**
		 * @brief Vertex position
		 */
		DEF_ARRAY_STATE(Coord, Position, DeviceType::GPU, "Vertex position");

		/**
		 * @brief Vertex velocity
		 */
		DEF_ARRAY_STATE(Coord, Velocity, DeviceType::GPU, "Vertex velocity");

	public:
		
		void loadSurface(std::string filename);

		//std::shared_ptr<Node> getSurface();

	protected:
		void resetStates() override;

		void postUpdateStates() override;
	};
}