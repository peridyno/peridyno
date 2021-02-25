#pragma once
#include "Framework/Node.h"
//#include "PointRenderModule.h"

namespace dyno
{
	template <typename TDataType> class TriangleSet;
	/*!
	*	\class	ParticleSystem
	*	\brief	Position-based fluids.
	*
	*	This class implements a position-based fluid solver.
	*	Refer to Macklin and Muller's "Position Based Fluids" for details
	*
	*/
	template<typename TDataType>
	class TriangularSurfaceMeshNode : public Node
	{
		DECLARE_CLASS_1(ParticleSystem, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		TriangularSurfaceMeshNode(std::string name = "TriangularSurfaceMeshNode");
		virtual ~TriangularSurfaceMeshNode();


		virtual bool translate(Coord t);
		virtual bool scale(Real s);

		DeviceArrayField<Coord>* getVertexPosition()
		{
			return &m_vertex_position;
		}

		DeviceArrayField<Coord>* getVertexVelocity()
		{
			return &m_vertex_velocity;
		}

		DeviceArrayField<Coord>* getVertexForce()
		{
			return &m_vertex_force;
		}

		DeviceArrayField<Triangle>* getTriangleIndex()
		{
			return &m_triangle_index;
		}

		std::shared_ptr<TriangleSet<TDataType>> getTriangleSet() { return m_triSet; }

		void updateTopology() override;
		bool resetStatus() override;

//		std::shared_ptr<PointRenderModule> getRenderModule();
	public:
		bool initialize() override;
//		virtual void setVisible(bool visible) override;

	protected:
		DeviceArrayField<Coord> m_vertex_position;
		DeviceArrayField<Coord> m_vertex_velocity;
		DeviceArrayField<Coord> m_vertex_force;
		DeviceArrayField<Triangle> m_triangle_index;

		std::shared_ptr<TriangleSet<TDataType>> m_triSet;
	};


#ifdef PRECISION_FLOAT
	template class TriangularSurfaceMeshNode<DataType3f>;
#else
	template class TriangularSurfaceMeshNode<DataType3d>;
#endif
}