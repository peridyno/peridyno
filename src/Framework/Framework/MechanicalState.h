#pragma once
#include <memory>
#include <vector>
#include <cassert>
#include <iostream>
#include <set>
#include "Platform.h"
#include "Framework/Module.h"
#include "Topology/FieldNeighbor.h"

namespace dyno
{
class MechanicalState : public Module
{
	DECLARE_CLASS(MechanicalState)

public:
	enum MaterialType {
		ParticleSystem = 0,
		FLUID,
		ELASTIC,
		PLASTIC,
		GRNULAR,
		UNDFINED
	};

public:
	MechanicalState();
	virtual ~MechanicalState(void);

	Real getTotalMass();
	void setTotalMass(Real mass);

	int getDOF();

	/**
	* @brief The following functions return the most commonly used Field IDs
	*/
	static FieldID position() { return "position"; }
	static FieldID pre_position() { return "pre_position"; }
	static FieldID init_position() { return "init_position"; }

	static FieldID velocity() { return "velocity"; }
	static FieldID pre_velocity() { return "pre_velocity"; }

	static FieldID angularVelocity() { return "angular_velocity"; }
	
	static FieldID acceleration() { return "acceleration"; }

	static FieldID force() { return "force"; }
	static FieldID torque() { return "force_moment"; }
	
	static FieldID mass() { return "mass"; }
	static FieldID angularMass() { return "angular_mass"; }
	static FieldID rotation() { return "rotation"; }


	/**
	* @brief The following functions return Field IDs for particle system
	*/
	static FieldID density() { return "density"; }
	static FieldID volume() { return "volume"; }
	static FieldID particle_neighbors() { return "particle_neighbors"; }
	static FieldID particle_attribute() { return "particle_attribute"; }
	static FieldID particle_normal() { return "particle_normal"; }


	static FieldID reference_particles() { return "reference_particles"; }

	std::string getModuleType() override { return "MechanicalState"; }

	MaterialType getMaterialType() { return m_type; }
	void setMaterialType(MaterialType type) { m_type = type; }

//	void resetForce();
//	void resetField(std::string name);

	/**
	* @brief The following functions operate on Field IDs that are specific to each numerical method
	*/
	FieldID addAuxiliaryID(FieldID id);
	void deleteAuxiliaryID(FieldID id);
	void clearAllIDs();
	bool hasAuxiliaryID(FieldID id);

	template<typename T>
	VarField<T>* allocVariable(std::string name, std::string description)
	{
		auto fd = new VarField<T>();
		bool ret = attachField(fd, name, description, true);
		if (!ret)
		{
			return nullptr;
		}

		return fd;
	}

	template<typename T>
	DeviceArrayField<T>* allocDeviceArray(std::string name, std::string description)
	{
		auto field = new DeviceArrayField<T>();
		bool ret = attachField(field, name, description, true);
		if (!ret)
		{
			return nullptr;
		}

		return field;
	}

	template<typename T>
	HostArrayField<T>* allocHostArray(std::string name, std::string description)
	{
		auto field = new HostArrayField<T>();
		bool ret = attachField(field, name, description, true);
		if (ret == false)
		{
			return nullptr;
		}

		return field;
	}

	
private:
	MaterialType m_type;

	/**m_auxIDs is used to store extra field IDs */
	std::set<FieldID> m_auxIDs;

	Real m_totalMass = 1.0f;
};
}