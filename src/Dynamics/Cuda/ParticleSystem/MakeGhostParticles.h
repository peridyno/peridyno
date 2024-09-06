#pragma once
#include "GhostParticles.h"
#include "Collision/Attribute.h"



namespace dyno
{
	template<typename TDataType>
	class MakeGhostParticles : public GhostParticles<TDataType>
	{
		DECLARE_TCLASS(MakeGhostParticles, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		MakeGhostParticles();
		~MakeGhostParticles() override;

		DEF_INSTANCE_IN(PointSet<TDataType>, Points, "");

		DEF_VAR(bool, ReverseNormal, true, "");


	protected:
		void resetStates() override;


	};


	IMPLEMENT_TCLASS(MakeGhostParticles, TDataType)
}