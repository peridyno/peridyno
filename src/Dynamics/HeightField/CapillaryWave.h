#pragma once
#include "ParticleSystem/ParticleSystem.h"
#include "Peridynamics/NeighborData.h"

namespace dyno
{
	/*!
	*	\class	CapillaryWave
	*	\brief	Peridynamics-based CapillaryWave.
	*/
	template<typename TDataType>
	class CapillaryWave : public Node
	{
		DECLARE_CLASS_1(CapillaryWave, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef TPair<TDataType> NPair;

		CapillaryWave(std::string name = "default");
		virtual ~CapillaryWave();

		bool translate(Coord t) override;
		bool scale(Real s) override;

		void loadSurface(std::string filename);

		std::shared_ptr<Node> getSurface();

	public:
		DEF_VAR(Real, Horizon, 0.01, "Horizon");

	protected:
		void resetStates() override;

		void updateTopology() override;

	private:
		std::shared_ptr<Node> mSurfaceNode;
	};
}