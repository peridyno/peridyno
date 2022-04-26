#include "Coupling.h"
#include "../Core/Matrix/Matrix3x3.h"
#include "../Core/Array/Array.h"

namespace dyno
{
	static void printFloat(float* device, int n)
	{
		float* host = new float[n];
		cudaMemcpy(host, device, n * sizeof(float), cudaMemcpyDeviceToHost);
		for (int i = 0; i < n; i++) {
			printf("a[%d] =%f\n", i, host[i]);
		}
		printf("end-----------------\n");
	}



	static void printfDArray2D(DArray2D<Vec4f> displacement) {
		int size = displacement.size();
		if (size == 0 ) {
			printf("size == 0");
			return;
		}
		CArray2D<Vec4f> host;
		host.resize(displacement.nx(),displacement.ny());

		host.assign(displacement);

		for (int i = 0; i < displacement.ny(); i++) {
			for(int j = 0; j < displacement.nx();j++)
			printf("[%d, %d]=%f %f %f %f\n", i,j, host[i,j].x, host[i, j].y, host[i, j].z, host[i, j].w);
		}

	}

	template<typename Coord>
	static void printfDArray(DArray<Coord> displacement) {
		int size = displacement.size();
		if (size == 0) {
			printf("size == 0");
			return;
		}
		CArray<Coord> host;
		host.resize(size);
		host.assign(displacement);
		for (int i = 0; i < size; i++) {
			printf("%d= %f %f %f\n",i,   host[i].x, host[i].y, host[i].z);
		}

	}
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
		cudaFree(m_forceX);
		cudaFree(m_forceY);
		cudaFree(m_forceZ);
		cudaFree(m_torqueX);
		cudaFree(m_torqueY);
		cudaFree(m_torqueZ);
		cudaFree(m_sample_heights);
	}

	template<typename TDataType>
	void Coupling<TDataType>::initialize()
	{
	
		int sizeInBytesF = 8 * sizeof(float);

		m_reduce = Reduction<float>::Create(8);
		m_force_corrected = false;
		cudaMalloc(&m_forceX, sizeInBytesF);
		cudaMalloc(&m_forceY, sizeInBytesF);
		cudaMalloc(&m_forceZ, sizeInBytesF);
		cudaMalloc(&m_torqueX, sizeInBytesF);
		cudaMalloc(&m_torqueY, sizeInBytesF);
		cudaMalloc(&m_torqueZ, sizeInBytesF);

		cudaMalloc(&m_sample_heights, sizeInBytesF);
		/*

		glm::vec3 center = boat->getCenter();


		float dg = m_trail->getRealGridSize();

		int nx = center.x / dg - m_trail->getGridSize() / 2;
		int ny = center.z / dg - m_trail->getGridSize() / 2;

		m_trail->setOriginX(nx);
		m_trail->setOriginY(ny);	*/
	}


	template<typename TDataType>
	void Coupling<TDataType>::resetStates()
	{


	}

	static int iDivUp(int a, int b)
	{
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}

	template<typename TDataType>
	void Coupling<TDataType>::updateStates()
	{
		this->animate(0.016f);
	}

	//__device__ int getDisplacement2(Vec3f pos, DArray2D<Vec4f> ocean, Vec2f origin, float patchSize, int gridSize)
	__device__ Vec4f getDisplacement2(Vec3f pos, DArray2D<Vec4f> oceanPatch, Vec2f origin, float patchSize, int gridSize)
	{
		Vec2f uv_i = (Vec2f(pos.x, pos.z) - origin) / patchSize;
		float u = (uv_i.x - floor(uv_i.x)) * gridSize;
		float v = (uv_i.y - floor(uv_i.y)) * gridSize;
		int i = floor(u);
		int j = floor(v);
		float fx = u - i;
		float fy = v - j;
		if (i == gridSize - 1)
		{
			i = gridSize - 2;
			fx = 1.0f;
		}
		if (j == gridSize - 1)
		{
			j = gridSize - 2;
			fy = 1.0f;
		}
		int id = i + j * gridSize;
		Vec4f d00 = oceanPatch[id];
		Vec4f d10 = oceanPatch[id + 1];
		Vec4f d01 = oceanPatch[id + gridSize];
		Vec4f d11 = oceanPatch[id + gridSize + 1];

		return d00 * (1 - fx) * (1 - fy) + d10 * fx * (1 - fy) + d01 * (1 - fx) * fy + d11 * fx * fy;
	}


	template<typename Matrix>
	__global__ void C_ComputeForceAndTorque(
		float* forceX,
		float* forceY,
		float* forceZ,
		float* torqueX,
		float* torqueY,
		float* torqueZ,
		float* sampleHeights,
		DArray<Vec3f> normals,
		DArray<Vec3f> samples,
		DArray2D<Vec4f> ocean,
		DArray<Vec3f> boatCenter,
		DArray<Matrix> rotation,
		Vec2f origin,
		int numOfSamples,
		float patchSize,
		int gridSize
	)
	{
		int pId = threadIdx.x + blockIdx.x * blockDim.x;
		if (pId < numOfSamples)
		{
			Vec3f dir_i = samples[pId];
			Vec3f rotDir = rotation[0] * Vec3f(dir_i.x, dir_i.y, dir_i.z);
			Vec3f pos_i = boatCenter[0] + rotation[0] * Vec3f(dir_i.x, dir_i.y, dir_i.z);

			//Vec4f dis_i = getDisplacement2(dir_i, ocean, origin, patchSize, gridSize);
			//getDisplacement----------------
			Vec2f uv_i = (Vec2f(pos_i.x, pos_i.z) - origin) / patchSize;
			float u = (uv_i.x - floor(uv_i.x)) * gridSize;
			float v = (uv_i.y - floor(uv_i.y)) * gridSize;
			int i = floor(u);
			int j = floor(v);
			float fx = u - i;
			float fy = v - j;
			if (i == gridSize - 1)
			{
				i = gridSize - 2;
				fx = 1.0f;
			}
			if (j == gridSize - 1)
			{
				j = gridSize - 2;
				fy = 1.0f;
			}
			int id = i + j * gridSize;
			Vec4f d00 = ocean[id];
			Vec4f d10 = ocean[id + 1];
			Vec4f d01 = ocean[id + gridSize];
			Vec4f d11 = ocean[id + gridSize + 1];

			Vec4f dis_i =  d00 * (1 - fx) * (1 - fy) + d10 * fx * (1 - fy) + d01 * (1 - fx) * fy + d11 * fx * fy;
			//-------------------------------


			dis_i.y *= 1.0f;
			Vec3f normal_i = normals[pId];
			Vec3f force_i(0,0,0);
			Vec3f torque_i(0, 0, 0);

			if (pos_i.y < dis_i.y)
			{
				force_i = 9800.0f * normal_i * (dis_i.y - pos_i.y);
				torque_i = Vec3f(0.0f, 9800.0f, 0.0f) * (dis_i.y - pos_i.y);
			}
			torque_i = rotDir.cross(torque_i);
			//torque_i = cross(Vec3f(rotDir.x, rotDir.y, rotDir.z), torque_i);
			forceX[pId] = force_i.x;
			forceY[pId] = force_i.y;
			forceZ[pId] = force_i.z;
			torqueX[pId] = torque_i.x;
			torqueY[pId] = torque_i.y;
			torqueZ[pId] = torque_i.z;
			sampleHeights[pId] = dis_i.y;
		}
	}

	template<typename TDataType>
	void Coupling<TDataType>::animate(float dt)
	{
		m_eclipsedTime += dt;

		//synchronCheck;
	
		auto m_boat = getRigidBodySystem();
		int num =8;
		int pDims = iDivUp(8, 64);

		auto center = getRigidBodySystem()->stateCenter()->getData();
		auto rotation = getRigidBodySystem()->stateRotationMatrix();
			
		//CArray<Vec3f> CVertexNormal;
		//CVertexNormal.resize(center.size());
		//CVertexNormal.assign(center);
		
		//获取正方体的顶点
		DArray<Coord> point = inTriangleSet()->getDataPtr()->getPoints();
		inTriangleSet()->getDataPtr()->updateVertexNormal();
		//printfDArray(point);
		auto VertexNormal = inTriangleSet()->getDataPtr()->outVertexNormal()->getData();
		//CArray<Vec3f> CVertexNormal;
		//CVertexNormal.resize(VertexNormal.size());
		//CVertexNormal.assign(VertexNormal);

		//获取海洋的高度
		DArray2D<Vec4f> Oceandisplacement = getOcean()->getOceanPatch()->getDisplacement();
		auto m_ocean_patch = getOcean()->getOceanPatch();
		//printfDArray2D(Oceandisplacement)
		/**/
		
		//计算力和力矩
		C_ComputeForceAndTorque << <pDims, 64 >> > (
			m_forceX,
			m_forceY,
			m_forceZ,
			m_torqueX,
			m_torqueY,
			m_torqueZ,
			m_sample_heights,
			VertexNormal,
			point,
			Oceandisplacement,
			m_boat->stateCenter()->getData(),//	m_boat->getCenter(), glm::vec3 boatCenter,   size == 1
			m_boat->stateRotationMatrix()->getData(),//m_boat->getOrientation(), glm::mat3 rotation,
			m_origin,
			8,
			m_ocean_patch->getPatchSize(),
			m_ocean_patch->getGridSize()
		);
		
		float fx = m_reduce->accumulate(m_forceX, 8);
		float fy = m_reduce->accumulate(m_forceY, 8);
		float fz = m_reduce->accumulate(m_forceZ, 8);

		float tx = m_reduce->accumulate(m_torqueX, 8);
		float ty = m_reduce->accumulate(m_torqueY, 8);
		float tz = m_reduce->accumulate(m_torqueZ, 8);

		float h = m_reduce->accumulate(m_sample_heights, 8);



	

		Vec3f force = Vec3f(fx / num, 0.0f, fz / num);
		Vec3f torque = Vec3f(tx / num, ty / num, tz / num);
		if (!m_force_corrected)
		{
			m_force_corrector = force;
			m_torque_corrector = torque;
			m_force_corrected = true;
		}

		
		//--------------------------------
		
		m_boat->updateVelocityAngule(force - m_force_corrector, torque - m_torque_corrector, dt);
		//	std::cout << "Angular Force: " << tx << " " << ty << " " << tz << std::endl;
		//m_boat->update(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(tx / num, ty / num, tz / num), dt);
		//m_boat->advect(dt);
		//--------------------------------
		//getRigidBodySystem()->stateCenter()->getData()
		DArray<Vec3f> m_center22 = m_boat->stateCenter()->getData();
		CArray<Vec3f> center22;
		center22.resize(1);
		center22.assign(m_center22);

		float m_heightShift = 0.25f;
		center22[0].y = h / 8 + m_heightShift;
		m_center22.assign(center22);
		m_boat->stateCenter()->setValue(m_center22);

		/**/
	/*	printFloat(m_forceX,8);
		printFloat(m_forceY,8);
		printFloat(m_forceZ,8);
		printFloat(m_torqueX,8);
		printFloat(m_torqueY,8);
		printFloat(m_torqueZ,8);*/



		printf("Coupling<TDataType>::animate  \n");
	}

	DEFINE_CLASS(Coupling);
}
