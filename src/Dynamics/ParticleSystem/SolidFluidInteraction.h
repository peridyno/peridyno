#pragma once
#include "Framework/Node.h"

namespace dyno
{
	template <typename T> class RigidBody;
	template <typename T> class ParticleSystem;
	template <typename T> class NeighborQuery;
	template <typename T> class DensityPBD;

	/*!
	*	\class	SolidFluidInteraction
	*	\brief	Position-based fluids.
	*
	*	This class implements a position-based fluid solver.
	*	Refer to Macklin and Muller's "Position Based Fluids" for details
	*
	*/

	template<typename TDataType>
	class SolidFluidInteraction : public Node
	{
		DECLARE_CLASS_1(SolidFluidInteraction, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SolidFluidInteraction(std::string name = "SolidFluidInteration");
		~SolidFluidInteraction() override;

	public:
		bool initialize() override;

		bool addRigidBody(std::shared_ptr<RigidBody<TDataType>> child);
		bool addParticleSystem(std::shared_ptr<ParticleSystem<TDataType>> child);

		bool resetStatus() override;

		void advance(Real dt) override;

		void setInteractionDistance(Real d);
	private:
		VarField<Real> radius;


		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Real> m_mass;
		DeviceArrayField<Coord> m_vels;


		DArray<int> m_objId;
		

		DArray<Coord> posBuf;
		DArray<Real> weights;
		DArray<Coord> init_pos;

		std::shared_ptr<NeighborList<int>> m_nList;
		std::shared_ptr<NeighborQuery<TDataType>> m_nbrQuery;

		std::shared_ptr<DensityPBD<TDataType>> m_pbdModule;;

		std::vector<std::shared_ptr<RigidBody<TDataType>>> m_rigids;
		std::vector<std::shared_ptr<ParticleSystem<TDataType>>> m_particleSystems;
	};
}