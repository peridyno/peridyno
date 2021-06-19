#pragma once
#include "ParticleSystem/ParticleSystem.h"

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

		Cloth(std::string name = "default");
		virtual ~Cloth();

		void advance(Real dt) override;

		void updateTopology() override;

		bool translate(Coord t) override;
		bool scale(Real s) override;

		bool initialize() override;

		void loadSurface(std::string filename);

		std::shared_ptr<Node> getSurface();

	private:
		std::shared_ptr<Node> mSurfaceNode;
	};
}