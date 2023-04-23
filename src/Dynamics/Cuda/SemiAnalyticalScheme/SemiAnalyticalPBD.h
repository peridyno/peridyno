/**
 * @author     : Yue Chang (yuechang@pku.edu.cn)
 * @date       : 2021-08-04
 * @description: Declaration of DensityPBDMesh class, which implements the position-based part of semi-analytical boundary conditions
 *               introduced in the paper <Semi-analytical Solid Boundary Conditions for Free Surface Flows>
 * @version    : 1.1
 */

#pragma once
#include "Module/ConstraintModule.h"
#include "Module/TopologyModule.h"

#include "ParticleSystem/Module/Kernel.h"

namespace dyno {
/**
 * DensityPBDMesh implements the position-based part of semi-analytical boundary conditions of the paper
 * <Semi-analytical Solid Boundary Conditions for Free Surface Flows>
 * It is used in PositionBasedFluidModelMesh class
 */
template <typename TDataType>
class SemiAnalyticalSummationDensity;

template <typename TDataType>
class SemiAnalyticalPBD : public ConstraintModule
{
    DECLARE_TCLASS(SemiAnalyticalPBD, TDataType)
public:
    typedef typename TDataType::Real          Real;
    typedef typename TDataType::Coord         Coord;
    typedef typename TopologyModule::Triangle Triangle;

    SemiAnalyticalPBD();
    ~SemiAnalyticalPBD() override;

    /**
     * handle the boundary conditions of fluids and mesh-based solid boundary
     * m_position&&m_velocity&&m_neighborhood&&m_neighborhoodTri&&Tri&&TriPoint need to be setup before calling this API
     *
     * @return true
     */
    void constrain() override;

    void takeOneIteration();

    void updateVelocity();

    void setIterationNumber(int n)
    {
        m_maxIteration = n;
    }

    DArray<Real>& getDensity()
    {
        return m_density.getData();
    }

public:
	DEF_VAR_IN(Real, TimeStep, "");

    FVar<Real> m_restDensity;

    /**
            * @brief smoothing length
            * A positive number represents the radius of neighborhood for each point
            */
	DEF_VAR_IN(Real, SmoothingLength, "");

	DEF_VAR_IN(Real, SamplingDistance, "");

    /**
	* @brief Particle position
	*/
	DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "");

    /**
	* @brief Particle velocity
	*/
	DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "");

    DeviceArrayField<Real>  m_massInv;

    /**
         * @brief neighbor list of particles, only neighbor pairs of particle-particle are counted
         */
    DEF_ARRAYLIST_IN(int, NeighborParticleIds, DeviceType::GPU, "");
    /**
         * @brief neighbor list of particles and mesh triangles, only neighbor pairs of particle-triangle are counted
         */
	DEF_ARRAYLIST_IN(int, NeighborTriangleIds, DeviceType::GPU, "");
    /**
         * @brief positions of Triangle vertexes
         */
	DEF_ARRAY_IN(Coord, TriangleVertex, DeviceType::GPU, "");

    /**
         * @brief Triangle indexes, represented by three integers, indicating the three indexes of triangle vertex
         */
	DEF_ARRAY_IN(Triangle, TriangleIndex, DeviceType::GPU, "");

    /**
         * @brief array of density, the output of DensitySummationMesh
         */
    DeviceArrayField<Real> m_density;

    FVar<int> use_mesh;
	FVar<int> use_ghost;

	FVar<int> Start;

private:
    int m_maxIteration;

    SpikyKernel<Real> m_kernel;

    DArray<Real>  m_lamda;
    DArray<Coord> m_deltaPos;
    DArray<Coord> m_position_old;

	std::shared_ptr<SemiAnalyticalSummationDensity<TDataType>> mCalculateDensity;
};

}  // namespace dyno