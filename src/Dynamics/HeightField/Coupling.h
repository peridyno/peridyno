#pragma once
#include <vector>
#include <iostream>
#include <string.h>

#include "CapillaryWave.h"
#include "OceanPatch.h"
#include "RigidBody\RigidBody.h"
namespace dyno
{
	template<typename TDataType>
	class Coupling : public Node
	{
		DECLARE_TCLASS(Coupling, TDataType)
	public:

		Coupling(std::string name = "");
		~Coupling();

		void initialize();

		void animate(float dt);

		void setHeightShift(float shift);

		void setBoatMatrix(glm::dmat4 mat, float dt);
		//glm::dmat4 getBoatMatrix();

		void steer(float degree);
		void propel(float acceleration);

		Vec2f getLocalBoatCenter();

		DEF_NODE_PORT(RigidBody<TDataType>, RigidBody, "RigidBody");
		DEF_NODE_PORT(CapillaryWave<TDataType>, CapillaryWave, "Capillary Wave");

	private:
		Vec2f m_origin;

		Vec3f m_prePos;

		std::string m_name;
		float m_heightShift;

		float m_eclipsedTime;

		//	float3* m_oceanCentroid;			//fft displacement at the center of boat

		float* m_forceX;					//forces at sample points
		float* m_forceY;
		float* m_forceZ;

		float* m_torqueX;					//torques at sample points
		float* m_torqueY;
		float* m_torqueZ;

		//采样点对应的海面高度
		float* m_sample_heights;

		bool m_force_corrected;

		Vec3f m_force_corrector;
		Vec3f m_torque_corrector;

		float m_heightScale = 0.2f;
	};
	IMPLEMENT_TCLASS(Coupling, TDataType)
}
