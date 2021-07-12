#pragma once
#include "ParticleSystem/ParticleSystem.h"
#include "NeighborData.h"

namespace dyno
{
	/*!
	*	\class	Cloth
	*	\brief	Peridynamics-based cloth.
	*/
	template<typename TDataType>
	class Cloth : public ParticleSystem<TDataType>
	{
		DECLARE_CLASS_1(Cloth, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef TPair<TDataType> NPair;

		Cloth(std::string name = "default");
		virtual ~Cloth();

		void updateTopology() override;

		bool resetStates() override;

		bool translate(Coord t) override;
		bool scale(Real s) override;

		bool initialize() override;

		void loadSurface(std::string filename);

		std::shared_ptr<Node> getSurface();

		DEF_VAR(Real, Horizon, 0.01, "Horizon");

		DEF_EMPTY_CURRENT_ARRAYLIST(NPair, RestShape, DeviceType::GPU, "Storing neighbors");

	private:
		std::shared_ptr<Node> mSurfaceNode;
	};
}