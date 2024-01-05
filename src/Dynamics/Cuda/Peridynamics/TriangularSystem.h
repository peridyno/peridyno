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
		DECLARE_TCLASS(TriangularSystem, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		TriangularSystem();
		~TriangularSystem() override;

		void addFixedParticle(int id, Coord pos);

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

		/**
		 * @brief Vertex velocity
		 */
		DEF_ARRAY_STATE(Coord, Force, DeviceType::GPU, "Vertex force");

	public:
		
		void loadSurface(std::string filename);

		//std::shared_ptr<Node> getSurface();

	protected:
		void updateTopology() override;
		void resetStates() override;

		std::shared_ptr<Node> mSurfaceNode;

		std::vector<int> m_fixedIds;
		std::vector<Coord> m_fixedPos;

		DeviceArrayField<int> FixedIds;
		DeviceArrayField<Coord> FixedPos;
//		virtual void setVisible(bool visible) override;
	};
}