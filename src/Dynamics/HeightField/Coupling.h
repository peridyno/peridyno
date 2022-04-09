#pragma once
#include <vector>
#include <iostream>
#include <string.h>

#include "CapillaryWave.h"
#include "OceanPatch.h"


template<typename TDataType>
class Coupling : public Node
{
	DECLARE_TCLASS(Coupling, TDataType)
public:

	Coupling(OceanPatch<TDataType> ocean_patch);
	~Coupling();

	void initialize(RigidBody<TDataType> boat, CapillaryWave<TDataType> wave);

	void animate(float dt);

	void setHeightShift(float shift);

	void setBoatMatrix(glm::dmat4 mat, float dt);
	glm::dmat4 getBoatMatrix();

	void steer(float degree);
	void propel(float acceleration);

	float2 getLocalBoatCenter();

	RigidBody<TDataType> getBoat();
	CapillaryWave<TDataType> getTrail();

	void setName(std::string name) { m_name = name; }


	DEF_NODE_PORT(RigidBody<TDataType>, RigidBody, "RigidBody");
	DEF_NODE_PORT(CapillaryWave<TDataType>,CapillaryWave, "Capillary Wave");
private:
	glm::vec3 m_prePos;

	std::string m_name;
	float m_heightShift;
	OceanPatch* m_ocean_patch;					//fft patch

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

	glm::vec3 m_force_corrector;
	glm::vec3 m_torque_corrector;

	float2 m_origin;

	float m_heightScale = 0.2f;

	IMPLEMENT_TCLASS(Coupling, TDataType)
};