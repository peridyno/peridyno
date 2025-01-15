
#pragma once
#include "ParticleSystem/Emitters/ParticleEmitter.h"

#include "Topology/EdgeSet.h"

namespace dyno
{
	template<typename TDataType>
	class Waterfall : public ParticleEmitter<TDataType>
	{
		DECLARE_TCLASS(Waterfall, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		Waterfall();
		virtual ~Waterfall();

		//void advance(Real dt) override;
	public:
		DEF_VAR(Real, Width, 0.1, "Emitter width");
		DEF_VAR(Real, Height, 0.1, "Emitter height");

		DEF_INSTANCE_STATE(EdgeSet<TDataType>, Outline, "Outline of the emitter");

	protected:
		void resetStates() override;

		void generateParticles() override;

	private:
		void tranformChanged();
	};

	IMPLEMENT_TCLASS(Waterfall, TDataType)
}