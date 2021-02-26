#pragma once
#include "Framework/Node.h"
#include "Framework/ModuleTopology.h"

#include "SemiAnalyticalIncompressibleFluidModel.h"
#include "PositionBasedFluidModelMesh.h"
#include "TriangularSurfaceMeshNode.h"

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
	class SemiAnalyticalSFINode : public Node
	{
		DECLARE_CLASS_1(SemiAnalyticalSFINode, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;


		SemiAnalyticalSFINode(std::string name = "SemiAnalyticalSFINode");
		~SemiAnalyticalSFINode() override;

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

//		DeviceArrayField<int> m_fixed;
//		DeviceArrayField<Coord> BoundryForce;
//		DeviceArrayField<Coord> ElasityForce;
//		DeviceArrayField<Real> ElasityPressure;


		GArray<int> m_objId;
		DeviceArrayField<int> ParticleId;

		GArray<Coord> posBuf;
		GArray<Coord> VelBuf;

		GArray<Real> weights;
		GArray<Coord> init_pos;
		
		//std::vector<std::shared_ptr<ParticleSystem<TDataType>>> m_particleSystems;
		//std::vector<std::shared_ptr<TriangularSurfaceMeshNode<TDataType>>> m_surfaces;
	};


#ifdef PRECISION_FLOAT
	template class SemiAnalyticalSFINode<DataType3f>;
#else
	template class SolidFluidInteractionTmp<DataType3d>;
#endif
}