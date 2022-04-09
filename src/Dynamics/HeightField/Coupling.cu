#include "Coupling.h"

namespace dyno
{

	template<typename TDataType>
	Coupling<TDataType>::Coupling(std::string name)
		: Node(name)
	{
		//printf("Coupling\n");

		m_origin = Vec2f(0, 0);
		m_force_corrected = false;

		m_heightShift = 6.0f;
		m_eclipsedTime = 0.0f;

		m_prePos = Vec3f(0, 0, 0);

		m_name = name;
	}

	template<typename TDataType>
	Coupling<TDataType>::~Coupling()
	{

	}
	template<typename TDataType>
	void Coupling<TDataType>::initialize()
	{


		/*
		int sizeInBytes = boat->getSamplingPointSize() * sizeof(float3);
		int sizeInBytesF = boat->getSamplingPointSize() * sizeof(float);

		m_reduce = Physika::Reduction<float>::Create(boat->getSamplingPointSize());

		cudaMalloc(&m_forceX, sizeInBytesF);
		cudaMalloc(&m_forceY, sizeInBytesF);
		cudaMalloc(&m_forceZ, sizeInBytesF);
		cudaMalloc(&m_torqueX, sizeInBytesF);
		cudaMalloc(&m_torqueY, sizeInBytesF);
		cudaMalloc(&m_torqueZ, sizeInBytesF);

		cudaMalloc(&m_sample_heights, sizeInBytesF);


		glm::vec3 center = boat->getCenter();


		float dg = m_trail->getRealGridSize();

		int nx = center.x / dg - m_trail->getGridSize() / 2;
		int ny = center.z / dg - m_trail->getGridSize() / 2;

		m_trail->setOriginX(nx);
		m_trail->setOriginY(ny);	*/
	}
	DEFINE_CLASS(Coupling);
}
