#pragma once

#include "Collision/Attribute.h"
#include "ParticleSystem/ParticleSystem.h"

#include "Peridynamics/TetrahedralSystem.h"

#include "Bond.h"
#include "EnergyDensityFunction.h"
#include "FilePath.h"

namespace dyno
{
	template<typename> class PointSetToPointSet;
	template<typename TDataType> class DistanceField3D;

	template<typename TDataType>
	class HyperelasticBody : public TetrahedralSystem<TDataType>
	{
		DECLARE_TCLASS(HyperelasticBody, TDataType)
	public:
		typedef typename TDataType::Real Real;

		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename TBond<TDataType> Bond;
		typedef typename TopologyModule::Tetrahedron Tetrahedron;
		
		HyperelasticBody();
		~HyperelasticBody() override;

		bool translate(Coord t);
		bool scale(Real s);
		bool scale(Coord s);

		bool rotate(Quat<Real> angle);

		bool rotate(Coord angle);

		void setEnergyModel(StVKModel<Real> model);
		void setEnergyModel(LinearModel<Real> model);
		void setEnergyModel(NeoHookeanModel<Real> model);
		void setEnergyModel(XuModel<Real> model);


		void loadSDF(std::string filename, bool inverted);
		//void updateStates() override;

	public:
		DEF_VAR(Vec3f, Location, 0, "Node location");
		DEF_VAR(Vec3f, Rotation, 0, "Node rotation");
		DEF_VAR(Vec3f, Scale, Vec3f(1.0f), "Node scale");

		DEF_VAR(Real, Horizon, 0.01, "Horizon");

		DEF_VAR(bool, AlphaComputed, true, "alphaComputed");

		DEF_VAR(EnergyType, EnergyType, NeoHooekean, "");

		DEF_VAR(EnergyModels<Real>, EnergyModel, EnergyModels<Real>(), "");

		DEF_ARRAY_STATE(Coord, RestPosition, DeviceType::GPU, "");

		DEF_ARRAYLIST_STATE(Bond, Bonds, DeviceType::GPU, "");

		DEF_ARRAYLIST_STATE(Real, VolumePair, DeviceType::GPU, "");


		DEF_ARRAY_STATE(Matrix, VertexRotation, DeviceType::GPU, "");

		DEF_ARRAY_STATE(Attribute, Attribute, DeviceType::GPU, "");

		DEF_ARRAY_STATE(Real, Volume, DeviceType::GPU, "");

		DEF_VAR(bool, NeighborSearchingAdjacent, true, "");

		DEF_VAR(FilePath, FileName, std::string(""), "");

		DEF_ARRAY_STATE(Tetrahedron, Tets, DeviceType::GPU, "");

		DEF_ARRAY_STATE(Real, DisranceSDF, DeviceType::GPU, "");

		//DEF_ARRAY_STATE(Coord, NormalSDF, DeviceType::GPU, "");


	protected:
		void resetStates() override;

		virtual void updateRestShape();
		virtual void updateVolume();
		std::shared_ptr<DistanceField3D<TDataType>> m_cSDF;
		DArray<Real> initDistance;
	};
}