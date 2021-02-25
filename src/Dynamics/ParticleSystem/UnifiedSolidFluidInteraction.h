#pragma once
#include "Framework/Node.h"
#include "Framework/ModuleTopology.h"

#include "UnifiedVelocityNode.h"
#include "PositionBasedFluidModelMesh.h"
#include "TriangularSurfaceMeshNode.h"
#include "UnifiedFluidRigidConstraint.h"
#include "RigidBody/RigidBodySystem.h"

namespace dyno
{
	class Attribute;
	template <typename T> class RigidBody;
	template <typename T> class ParticleSystem;
	template <typename T> class TriangularSurfaceMeshNode;
	template <typename T> class NeighborQuery;
	template <typename TDataType> class PointSet;
	template <typename TDataType> class TriangleSet;
	/*!
	*	\class	SolidFluidInteraction
	*	\brief	Position-based fluids.
	*
	*	This class implements a position-based fluid solver.
	*	Refer to Macklin and Muller's "Position Based Fluids" for details
	*
	*/

	template<typename TDataType>
	class UnifiedSolidFluidInteraction : public Node
	{
		DECLARE_CLASS_1(UnifiedSolidFluidInteraction, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;
		typedef typename TDataType::Matrix Matrix;

		UnifiedSolidFluidInteraction(std::string name = "UnifiedSolidFluidInteraction");
		~UnifiedSolidFluidInteraction() override;

		void setRigidModule(std::shared_ptr<RigidBodySystem<TDataType>> r)
		{
			m_rigidModule = r;
		}

	public:
		bool initialize() override;

		//std::shared_ptr<Node> addChild(std::shared_ptr<Node> child) override;

		//bool addParticleSystem(std::shared_ptr<ParticleSystem<TDataType>> child);
		//bool addTriangularSurfaceMesh(std::shared_ptr<TriangularSurfaceMeshNode<TDataType>> child);

		bool resetStatus() override;

		void advance(Real dt) override;

		void setInteractionDistance(Real d);


		DeviceArrayField<Coord>* getParticlePosition()
		{
			return &m_particle_position;
		}

		DeviceArrayField<Attribute>* getParticleAttribute()
		{
			return &m_particle_attribute;
		}
		DeviceArrayField<Coord>* getParticleVelocity()
		{
			return &m_particle_velocity;
		}

		DeviceArrayField<Coord>* getParticleForceDensity()
		{
			return &m_particle_force_density;
		}

		DeviceArrayField<Real>* getParticleMass()
		{
			return &m_particle_mass;
		}

		// 		DeviceArrayField<int>* getParticleId()
		// 		{
		// 			return &ParticleId;
		// 		}

		DeviceArrayField<Coord>* getTriangleVertex()
		{
			return &m_triangle_vertex;
		}
		DeviceArrayField<Coord>* getTPO()
		{
			return &m_triangle_vertex_old;
		}
		DeviceArrayField<Triangle>* getTriangleIndex()
		{
			return &m_triangle_index;
		}
		DeviceArrayField<Real>* getTriangleVertexMass()
		{
			return &m_triangle_vertex_mass;
		}

	private:

		DEF_NODE_PORTS(ParticleSystem, ParticleSystem<TDataType>, "Particle Systems");
		DEF_NODE_PORTS(TriangularSurfaceMeshNode, TriangularSurfaceMeshNode<TDataType>, "Triangular Surface Mesh Node");


		VarField<Real> radius;

		DeviceArrayField<Real> m_particle_mass;
		DeviceArrayField<Coord> m_particle_velocity;
		DeviceArrayField<Coord> m_particle_position;
		DeviceArrayField<Attribute> m_particle_attribute;

		DeviceArrayField<Real> m_triangle_vertex_mass;
		DeviceArrayField<Coord> m_triangle_vertex;
		DeviceArrayField<Coord> m_triangle_vertex_old;
		DeviceArrayField<Triangle> m_triangle_index;

		DeviceArrayField<Coord> m_particle_force_density;

		

		DeviceArrayField<Real> m_boundary_pressure;//
		DeviceArrayField<Real> m_gradient_point;//
		DeviceArrayField<NeighborConstraints> m_nbrcons;//
		DeviceArrayField<Real> m_gradient_rigid;//
		DeviceArrayField<Coord> m_velocity_inside_iteration;//

		DeviceArrayField<Real> m_rigid_mass;
		DeviceArrayField<Matrix> m_rigid_interior;//
		DeviceArrayField<Coord> AA;//
		DeviceArrayField<Coord> m_rigid_velocity;//
		DeviceArrayField<Coord> m_rigid_angular_velocity;//
		DeviceArrayField<Coord> m_rigid_position;//
		DeviceArrayField<Matrix> m_rigid_rotation;//


		DeviceArray<int> m_objId;
		DeviceArrayField<int> ParticleId;

		DeviceArray<Coord> posBuf;
		DeviceArray<Coord> VelBuf;

		DeviceArray<Real> weights;
		DeviceArray<Coord> init_pos;


		std::shared_ptr<UnifiedFluidRigidConstraint<TDataType>> m_intermediateModule;
		std::shared_ptr<RigidBodySystem<TDataType>> m_rigidModule;
		std::shared_ptr<UnifiedVelocityNode<DataType3f>> m_fluidModule;

		//std::vector<std::shared_ptr<ParticleSystem<TDataType>>> m_particleSystems;
		//std::vector<std::shared_ptr<TriangularSurfaceMeshNode<TDataType>>> m_surfaces;
	};


#ifdef PRECISION_FLOAT
	template class UnifiedSolidFluidInteraction<DataType3f>;
#else
	template class UnifiedSolidFluidInteraction<DataType3d>;
#endif
}