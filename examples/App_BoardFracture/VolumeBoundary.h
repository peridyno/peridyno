#pragma once
#include "Volume/Volume.h"
#include "ParticleSystem/ParticleSystem.h"

namespace dyno {

	template<typename TDataType>
	class VolumeBoundary : public Volume<TDataType>
	{
		DECLARE_CLASS_1(VolumeBoundary, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VolumeBoundary();
		~VolumeBoundary() override;

		void advance(Real dt) override;

		void translate(Coord t);

		std::shared_ptr<Node> loadSDF(std::string filename, bool bOutBoundary = false);
		std::shared_ptr<Node> loadCube(Coord lo, Coord hi, Real distance = 0.005f, bool bOutBoundary = false);

// 		void loadSDF(std::string filename, bool bOutBoundary = false);
// 		std::shared_ptr<Node> loadCube(Coord lo, Coord hi, Real distance = 0.005f, bool bOutBoundary = false);
// 		void loadShpere(Coord center, Real r, Real distance = 0.005f, bool bOutBoundary = false, bool bVisible = false);


	public:
		DEF_EMPTY_VAR(TangentialFriction, Real, "Tangential friction");
		DEF_EMPTY_VAR(NormalFriction, Real, "Normal friction");

		DEF_NODE_PORTS(ParticleSystem, ParticleSystem<TDataType>, "Particle Systems");

	private:
		std::shared_ptr<Node> m_surfaceNode;
	};


#ifdef PRECISION_FLOAT
template class VolumeBoundary<DataType3f>;
#else
template class VolumeBoundary<DataType3d>;
#endif

}
