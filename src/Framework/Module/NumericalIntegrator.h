#pragma once
#include "Module.h"

namespace dyno
{
	class NumericalIntegrator : public Module
	{
	public:
		NumericalIntegrator();
		~NumericalIntegrator() override;

		virtual void begin() {};
		virtual void end() {};

		virtual bool integrate() { return true; }

		void setMassID(FieldID id) { m_massID = id; }
		void setForceID(FieldID id) { m_forceID = id; }
		void setTorqueID(FieldID id) { m_torqueID = id; }
		void setPositionID(FieldID id) { m_posID = id; }
		void setVelocityID(FieldID id) { m_velID = id; }
		void setPositionPreID(FieldID id) { m_posPreID = id; }
		void setVelocityPreID(FieldID id) { m_velPreID = id; }

		std::string getModuleType() override { return "NumericalIntegrator"; }

	protected:
		FieldID m_massID;
		FieldID m_forceID;
		FieldID m_torqueID;
		FieldID m_posID;
		FieldID m_velID;
		FieldID m_posPreID;
		FieldID m_velPreID;
	};
}

