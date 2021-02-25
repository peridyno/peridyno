#pragma once
#include "ParticleSystem/ParticleSystem.h"

namespace dyno
{
	/*!
	*	\class	ParticleCloth
	*	\brief	Peridynamics-based elastic object.
	*/
	template<typename TDataType>
	class ParticleCloth : public ParticleSystem<TDataType>
	{
		DECLARE_CLASS_1(ParticleCloth, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleCloth(std::string name = "default");
		virtual ~ParticleCloth();

		void advance(Real dt) override;

		void updateTopology() override;

		bool translate(Coord t) override;
		bool scale(Real s) override;

		bool initialize() override;

		void loadSurface(std::string filename);

	private:
		std::shared_ptr<Node> m_surfaceNode;
	};

#ifdef PRECISION_FLOAT
	template class ParticleCloth<DataType3f>;
#else
	template class ParticleCloth<DataType3d>;
#endif
}