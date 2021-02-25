#pragma once
#include "ParticleSystem.h"

namespace dyno
{
	/*!
	*	\class	ParticleFluid
	*	\brief	Position-based fluids.
	*
	*	This class implements a position-based fluid solver.
	*	Refer to Macklin and Muller's "Position Based Fluids" for details
	*
	*/
	class Attribute;
	class SurfaceMeshRender;
	template<typename TDataType>
	class ParticleFluidMesh : public ParticleSystem<TDataType>
	{
		DECLARE_CLASS_1(ParticleFluidMesh, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleFluidMesh(std::string name = "fluid");
		virtual ~ParticleFluidMesh();

		void advance(Real dt) override;

		void loadSurface(std::string filename);
		std::shared_ptr<SurfaceMeshRender> getSurfaceRender() { return m_surfaceRender; }
		std::shared_ptr<Node> getSurfaceNode() { return m_surfaceNode; }

		DeviceArrayField<Attribute> m_attribute;
		DeviceArrayField<Coord> m_position2;
		DeviceArrayField<Coord> m_normal2;

		
		
	private:
		std::shared_ptr<Node> m_surfaceNode;
		std::shared_ptr<SurfaceMeshRender> m_surfaceRender;
	};

#ifdef PRECISION_FLOAT
	template class ParticleFluidMesh<DataType3f>;
#else
	template class ParticleFluidMesh<DataType3d>;
#endif
}