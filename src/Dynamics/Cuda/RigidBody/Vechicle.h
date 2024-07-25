#pragma once
#include "Node/ParametricModel.h"
#include "RigidBodySystem.h"

#include "Topology/TextureMesh.h"
#include "Topology/TriangleSet.h"

#include "STL/Pair.h"
#include "VehicleInfo.h"

namespace dyno 
{
	template<typename TDataType>
	class Vechicle : virtual public ParametricModel<TDataType>, virtual public RigidBodySystem<TDataType>
	{
		DECLARE_TCLASS(Vechicle, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename Pair<uint, uint> BindingPair;

		Vechicle();
		~Vechicle() override;

		void bind(std::shared_ptr<PdActor> actor, Pair<uint, uint> shapeId);

	public:
		DEF_INSTANCE_IN(TextureMesh, TextureMesh, "Texture mesh of the vechicle");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "TriangleSet of the boundary");

	public:
		DEF_ARRAYLIST_STATE(Transform3f, InstanceTransform, DeviceType::GPU, "Instance transforms");

	protected:
		void resetStates() override;

		void updateStates() override;

		void updateInstanceTransform();

		void clearVechicle();

		void transform();

	protected:

		DArray<Matrix> mInitialRot;

	private:
		std::vector<Pair<uint, uint>> mBindingPair;

		std::vector<std::shared_ptr<PdActor>> mActors;

		DArray<BindingPair> mBindingPairDevice;

		DArray<int> mBindingTagDevice;
	};





	template<typename TDataType>
	class Jeep : virtual public Vechicle<TDataType>
	{
		DECLARE_TCLASS(Jeep, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		Jeep();
		~Jeep() override;

	protected:
		void resetStates() override;
	};


	template<typename TDataType>
	class ConfigurableVehicle : virtual public Vechicle<TDataType>
	{
		DECLARE_TCLASS(ConfigurableVehicle, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ConfigurableVehicle();
		~ConfigurableVehicle() override;

		DEF_VAR(VehicleBind,VehicleConfiguration, VehicleBind(4), "");

	protected:
		void resetStates() override;
	};

}
