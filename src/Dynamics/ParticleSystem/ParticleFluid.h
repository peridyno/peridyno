#pragma once
#include "ParticleSystem.h"
#include "ParticleEmitter.h"

namespace dyno
{
	template<typename TDataType>
	class ParticleFluid : public ParticleSystem<TDataType>
	{
		DECLARE_CLASS_1(ParticleFluid, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleFluid(std::string name = "default");
		virtual ~ParticleFluid();

		bool resetStates() override;

	protected:
		void preUpdate();

	private:
		DEF_NODE_PORTS(ParticleEmitter, ParticleEmitter<TDataType>, "Particle Emitters");
	};
}