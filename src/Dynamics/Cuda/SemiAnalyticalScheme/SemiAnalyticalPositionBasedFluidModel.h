/**
 * @author     : Yue Chang (yuechang@pku.edu.cn)
 * @date       : 2021-08-06
 * @description: Declaration of PositionBasedFluidModelMesh class, a container for semi-analytical PBD fluids 
 *               introduced in the paper <Semi-analytical Solid Boundary Conditions for Free Surface Flows>
 * @version    : 1.1
 */
#pragma once
#include "SemiAnalyticalPBD.h"

#include "Module/TopologyModule.h"
#include "Module/GroupModule.h"
#include "TriangularMeshConstraint.h"

#include "Collision/Attribute.h"

/**
 * PositionBasedFluidModelMesh
 * a NumericalModel for semi-analytical PBD fluids 
 * The solver is PBD fluids with semi-analytical boundaries
 * reference: "Semi-analytical Solid Boundary Conditions for Free Surface Flows"
 *
 * Could be used by being created and initialized at SemiAnalyticalSFINode
 * Fields required to be initialized include:
 *     m_position
 *     m_velocity
 *     m_forceDensity
 *     m_vn
 *     TriPoint
 *     TriPointOld
 *     Tri
 *     m_smoothingLength
 * 
 *
 */

namespace dyno 
{
	template <typename TDataType>
	class SemiAnalyticalPositionBasedFluidModel : public GroupModule
	{
		DECLARE_TCLASS(SemiAnalyticalPositionBasedFluidModel, TDataType)
	public:
		typedef typename TDataType::Real  Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		SemiAnalyticalPositionBasedFluidModel();
		virtual ~SemiAnalyticalPositionBasedFluidModel() {};

	public:
		FVar<int> Start;

		DEF_VAR(Real, SmoothingLength, 0.01, "Smoothing length");

		DEF_VAR_IN(Real, TimeStep, "Time step size");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "");	//current particle position
		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "");  //current particle velocity
		DEF_ARRAY_IN(Coord, Force, DeviceType::GPU, "");


		DEF_ARRAY_IN(Coord, TriangleVertex, DeviceType::GPU, "");     //triangle vertex point position
		DEF_ARRAY_IN(Triangle, TriangleIndex, DeviceType::GPU, "");          //triangle index
	};
}  // namespace PhysIKA