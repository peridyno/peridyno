#pragma once
#include "Node.h"

#include "Topology/EdgeSet.h"

namespace dyno
{
	/*!
	*	\class	TetSystem
	*	\brief	This class represents the base class for more advanced particle-based nodes.
	*/
	template<typename TDataType>
	class ThreadSystem : public Node
	{
		DECLARE_TCLASS(ThreadSystem, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ThreadSystem();
		~ThreadSystem() override;

		
		DEF_INSTANCE_STATE(EdgeSet<TDataType>, EdgeSet, "");

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
		
		/*void loadVertexFromFile(std::string filename);

		void loadVertexFromGmshFile(std::string filename);*/

		void addThread(Coord start, Coord end, int segSize);


	protected:
		void updateTopology() override;
		void resetStates() override;

		std::vector<Coord> particles;
		std::vector<TopologyModule::Edge> edges;

//		virtual void setVisible(bool visible) override;
	};
}