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
		DECLARE_TCLASS(Cloth, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef TPair<TDataType> NPair;

		Cloth(std::string name = "default");
		virtual ~Cloth();

		bool translate(Coord t) override;
		bool scale(Real s) override;

		void loadSurface(std::string filename);

		std::shared_ptr<Node> getSurface();

	public:
		DEF_VAR(Real, Horizon, 0.01, "Horizon");

		DEF_ARRAYLIST_STATE(NPair, RestShape, DeviceType::GPU, "Storing neighbors");

	protected:
		void resetStates() override;

		void updateTopology() override;

	private:
		std::shared_ptr<Node> mSurfaceNode;
	};
}