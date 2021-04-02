#pragma once
#include "Framework/CollisionModel.h"
#include "Framework/ModuleTopology.h"
#include "Framework/Node.h"

namespace dyno
{
	template <typename> class CollidablePoints;
	template <typename> class NeighborQuery;
	template <typename> class NeighborList;
	template <typename> class GridHash;
	template <typename TDataType> class PointSet;
	template <typename TDataType> class TriangleSet;

	template<typename TDataType>
	class MeshCollision : public CollisionModel
	{
		DECLARE_CLASS_1(MeshCollision, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		MeshCollision();
		virtual ~MeshCollision();

		bool isSupport(std::shared_ptr<CollidableObject> obj) override;

		void addCollidableObject(std::shared_ptr<CollidableObject> obj) override;

		bool initializeImpl() override;

		void doCollision() override;


		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Coord> m_velocity;
		DeviceArrayField<Real> m_triangle_vertex_mass;
		DeviceArrayField<Coord> m_triangle_vertex;
		DeviceArrayField<Coord> m_triangle_vertex_old;
		DeviceArrayField<Triangle> m_triangle_index;
		DeviceArrayField<int> m_flip;
		NeighborField<int> m_neighborhood_tri;

		DeviceArrayField<Coord> m_velocity_mod;



	protected:
		DArray<int> m_objId;

		DArray<Real> weights;
		DArray<Coord> init_pos;
		DArray<Coord> posBuf;

		DArray<Coord> m_position_previous;
		DArray<Coord> m_triangle_vertex_previous;

		std::shared_ptr<NeighborQuery<TDataType>> m_nbrQuery;
		std::shared_ptr<NeighborList<int>> m_nList;

		std::vector<std::shared_ptr<CollidablePoints<TDataType>>> m_collidableObjects;

	};
}
