#pragma once
#include "Module/GroupModule.h"
#include "Module/TopologyModule.h"

namespace dyno
{
	/*!
	*	\class	SemiAnalyticalSurfaceTensionModel
	*	\brief	Semi-Analytical Surface Tension Model for Free Surface Flows
	*
	*	This class encapsulates all necessary modules to implement a semi-analytical solution for free surface flows with surface surface.
	*	Refer to Menglik et al.s "Semi-Analytical Surface Tension Model for Free Surface Flows", IEEE VR, Poster, 2022 for details
	*
	*/
	typedef typename TopologyModule::Triangle Triangle;
	class Attribute;

	template<typename TDataType>
	class SemiAnalyticalSurfaceTensionModel : public GroupModule
	{
		DECLARE_TCLASS(SemiAnalyticalSurfaceTensionModel, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		
		SemiAnalyticalSurfaceTensionModel();
		virtual ~SemiAnalyticalSurfaceTensionModel() {};

	public:

		DEF_VAR(Real, SmoothingLength, Real(0.006), "smoothing length");//0.006
		DEF_VAR_IN(Real, TimeStep, "Time step size!");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "");
		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "");
		DEF_ARRAY_IN(Coord, ForceDensity, DeviceType::GPU, "");

		DEF_ARRAY_IN(Attribute, Attribute, DeviceType::GPU, "Particle attribute");

		DEF_ARRAY_IN(Triangle, TriangleInd, DeviceType::GPU, "triangle_index");
		DEF_ARRAY_IN(Coord, TriangleVer, DeviceType::GPU, "triangle_vertex");

		DEF_VAR(Real, SurfaceTension, Real(0.055), "surface tension");
		DEF_VAR(Real, AdhesionIntensity, Real(30.0), "adhesion");
		DEF_VAR(Real, RestDensity, Real(1000), "Rest Density");

		DeviceArrayField<int> m_flip;
		DeviceArrayField<Coord> m_velocity_mod;
	};
}