/**
 * @author     : Yue Chang (yuechang@pku.edu.cn)
 * @date       : 2021-08-04
 * @description: Declaration of SemiAnalyticalPBD class, which implements the position-based part of semi-analytical boundary conditions
 *               introduced in the paper <Semi-analytical Solid Boundary Conditions for Free Surface Flows>
 * @version    : 1.1
 */

#pragma once
#include "Module/ConstraintModule.h"
#include "Module/TopologyModule.h"

#include "ParticleSystem/Module/ParticleApproximation.h"

namespace dyno
{
    /**
     * SemiAnalyticalPBD implements the position-based part of semi-analytical boundary conditions of the paper
     * <Semi-analytical Solid Boundary Conditions for Free Surface Flows>
     * It is used in SemiAnalyticalPositionBasedFluidModel class
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
         */
        void constrain() override;

    public:
        DEF_VAR(uint, InterationNumber, 3, "");

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

    private:
        void takeOneIteration();

        void updateVelocity();

        SpikyKernel<Real> m_kernel;

        DArray<Real>  mLamda;
        DArray<Coord> mDeltaPos;
        DArray<Coord> mPosBuffer;

        std::shared_ptr<SemiAnalyticalSummationDensity<TDataType>> mCalculateDensity;
    };
}  // namespace dyno