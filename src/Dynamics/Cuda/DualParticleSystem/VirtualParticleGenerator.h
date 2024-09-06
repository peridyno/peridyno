/*
*@Brief: Base class for the virtual particle generator in Dual-particle SPH method.
*@Paper: Liu et al., ACM Trans Graph (TOG). 2024. (A Dual-Particle Approach for Incompressible SPH Fluids) doi.org/10.1145/3649888
*/

#pragma once
#include "Module/ConstraintModule.h"

namespace dyno 
{
	template<typename TDataType>
	class VirtualParticleGenerator : public ConstraintModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VirtualParticleGenerator();
		~VirtualParticleGenerator() override {};

		/**
		* @brief Virtual Particle positions
		*/
		DEF_ARRAY_OUT(Coord, VirtualParticles, DeviceType::GPU, "Output virtual particle position");
	};
}