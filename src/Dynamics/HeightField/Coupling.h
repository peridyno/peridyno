#pragma once
#include <vector>
#include <iostream>
#include <string.h>

#include "CapillaryWave.h"
#include "OceanPatch.h"
#include "Ocean.h"
#include "RigidBody\RigidBodySystem.h"
#include "Topology/TriangleSet.h"
#include "../Core/Algorithm/Reduction.h"
namespace dyno
{
	template<typename TDataType>
	class Coupling : public Node
	{
		DECLARE_TCLASS(Coupling, TDataType)
	public:

		typedef typename TDataType::Matrix Matrix;
		typedef typename TDataType::Coord Coord;

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

		DEF_NODE_PORT(RigidBodySystem<TDataType>, RigidBodySystem, "RigidBodySystem");
		DEF_NODE_PORT(Ocean<TDataType>, Ocean, "Ocean");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "");
		
	protected:
		void resetStates() override;
		void updateStates() override;

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

		Reduction<float>* m_reduce;
	};
	IMPLEMENT_TCLASS(Coupling, TDataType)
}
