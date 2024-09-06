#include "VolumeBool.h"
#include <random>
#include <iostream>
#include <ctime>
#define FarDistance 100000
namespace dyno
{
	IMPLEMENT_TCLASS(VolumeBool, TDataType)
		
	template<typename Real>
	__global__ void C_FastMarching(
		DArray3D<Real> bigDistance,
		DArray3D<Real> m_distance,
		float h,
		DArray<int> pointType) {

		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		uint nx = bigDistance.nx();
		uint ny = bigDistance.ny();
		uint nz = bigDistance.nz();

		//碰到bigDistance边界返回
		if (i == 0 || j == 0 || k == 0 ||
			i >= nx - 1 || j >= ny - 1 || k >= nz - 1) return;

		int pointIndex = i - 1 + (j - 1) * (nx - 2) + (k - 1) * (nx - 2) * (ny - 2);
		
		//如果是已知的点就跳过
		//if (pointType[pointIndex] == 1) return;

		//fast marching 核心算法
		Real Ux, Uy, Uz;
		if (m_distance(i - 1, j - 1, k - 1) < 0)
		{
			if (bigDistance(i - 1, j, k) < 0 && bigDistance(i + 1, j, k) < 0)
				Ux = maximum(bigDistance(i - 1, j, k), bigDistance(i + 1, j, k));
			else
				Ux = minimum(bigDistance(i - 1, j, k), bigDistance(i + 1, j, k));
			if (bigDistance(i, j - 1, k) < 0 && bigDistance(i, j + 1, k) < 0)
				Uy = maximum(bigDistance(i, j - 1, k), bigDistance(i, j + 1, k));
			else
				Uy = minimum(bigDistance(i, j - 1, k), bigDistance(i, j + 1, k));
			if (bigDistance(i, j, k - 1) < 0 && bigDistance(i, j, k + 1) < 0)
				Uz = maximum(bigDistance(i, j, k - 1), bigDistance(i, j, k + 1));
			else
				Uz = minimum(bigDistance(i, j, k - 1), bigDistance(i, j, k + 1));
		}
		else
		{
			if (bigDistance(i - 1, j, k) > 0 && bigDistance(i + 1, j, k) > 0)
				Ux = minimum(bigDistance(i - 1, j, k), bigDistance(i + 1, j, k));
			else
				Ux = maximum(bigDistance(i - 1, j, k), bigDistance(i + 1, j, k));
			if (bigDistance(i, j - 1, k) > 0 && bigDistance(i, j + 1, k) > 0)
				Uy = minimum(bigDistance(i, j - 1, k), bigDistance(i, j + 1, k));
			else
				Uy = maximum(bigDistance(i, j - 1, k), bigDistance(i, j + 1, k));
			if (bigDistance(i, j, k - 1) > 0 && bigDistance(i, j, k + 1) > 0)
				Uz = minimum(bigDistance(i, j, k - 1), bigDistance(i, j, k + 1));
			else
				Uz = maximum(bigDistance(i, j, k - 1), bigDistance(i, j, k + 1));
		}

		//旁边没有 know节点
		if (abs(Ux) == FarDistance && abs(Uy) == FarDistance && abs(Uz) == FarDistance) return;

		Real x1 = maximum(maximum(Ux, Uy), Uz);
		Real x3 = minimum(minimum(Ux, Uy), Uz);
		Real x2;
		if (Ux < x1 && Ux > x3)x2 = Ux;
		else if(Uy < x1 && Uy > x3)x2 = Uy;
		else x2 = Uz;

		// Ux > Uy > Uz
		Ux = x1;
		Uy = x2;
		Uz = x3;

		Real value;
		if (m_distance(i - 1, j - 1, k - 1) < 0)
		{
			Real s1 = (Ux + Uy + Uz - sqrt(3 * h * h - (Ux - Uy) * (Ux - Uy) - (Ux - Uz) * (Ux - Uz) - (Uy - Uz) * (Uy - Uz))) / 3;
			Real s11 = 3 * h * h - (Ux - Uy) * (Ux - Uy) - (Ux - Uz) * (Ux - Uz) - (Uy - Uz) * (Uy - Uz);
			Real s2 = (Uy + Ux - sqrt(2 * h * h - (Uy - Ux) * (Uy - Ux))) / 2;
			Real s22 = 2 * h * h - (Uy - Ux) * (Uy - Ux);
			if (s1 < Uz)
				value = s1;
			else if ((s2 < Uy) && (s2 > Uz))
				value = s2;
			else
				value = Ux - h;

			bigDistance(i, j, k) = max(bigDistance(i, j, k), -abs(value));
		}
		else
		{
			Real s1 = (Ux + Uy + Uz + sqrt(3 * h * h - (Ux - Uy) * (Ux - Uy) - (Ux - Uz) * (Ux - Uz) - (Uy - Uz) * (Uy - Uz))) / 3;
			Real s11 = 3 * h * h - (Ux - Uy) * (Ux - Uy) - (Ux - Uz) * (Ux - Uz) - (Uy - Uz) * (Uy - Uz);
			Real s2 = (Uy + Uz + sqrt(2 * h * h - (Uy - Uz) * (Uy - Uz))) / 2;
			Real s22 = 2 * h * h - (Uy - Uz) * (Uy - Uz);
			if (s1 > Ux)
				value = s1;
			else if ((s2 < Ux) && (s2 > Uy))
				value = s2;
			else
				value = Uz + h;

			bigDistance(i, j, k) = min(bigDistance(i, j, k), abs(value));
		}

		pointType[pointIndex] = 1;
	}

	template<typename TDataType, typename Real, typename Coord>
	__global__ void C_CopySDFValue(
		DArray3D<Real> bigDistance,
		DArray3D<Real> m_distance,
		DArray<int> pointType,
		DistanceField3D<TDataType> aDistance,
		DistanceField3D<TDataType> bDistance,
		Coord m_h,
		Coord m_left) {

		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		uint nx = bigDistance.nx();
		uint ny = bigDistance.ny();
		uint nz = bigDistance.nz();

		if (i >= nx || j >= ny || k >= nz) return;

		//碰到bigDistance边界返回
		if (i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1)
		{
			bigDistance(i, j, k) = FarDistance;
			return;
		}

		if (m_distance(i - 1, j - 1, k - 1) < 0)
			bigDistance(i, j, k) = -FarDistance;//设置far的距离
		else
			bigDistance(i, j, k) = FarDistance;//设置far的距离
		pointType[i - 1 + (j - 1) * m_distance.nx() + (k - 1) * m_distance.nx() * m_distance.ny()] = 0;

		Coord point = m_left + Coord((i - 1) * m_h[0], (j - 1) * m_h[1], (k - 1) * m_h[2]);
		Real a;
		Coord normal;
		aDistance.getDistance(point, a, normal);
		Real b;
		bDistance.getDistance(point, b, normal);

		if (abs(m_distance(i - 1, j - 1, k - 1)) < 0.2)
		{
			if ((abs(a) < 0.2&&b > 0) || (abs(b) < 0.2&&a > 0))
			{
				bigDistance(i, j, k) = m_distance(i - 1, j - 1, k - 1);
				pointType[i - 1 + (j - 1) * m_distance.nx() + (k - 1) * m_distance.nx() * m_distance.ny()] = 1;
			}
		}
	}
	template<typename Real>
	__global__ void C_GetSDFValue(
		DArray3D<Real> m_distance,
		DArray3D<Real> bigDistance) {

		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		uint nx = m_distance.nx();
		uint ny = m_distance.ny();
		uint nz = m_distance.nz();

		if (i >= nx || j >= ny || k >= nz) return;
	
		m_distance(i, j, k) = bigDistance(i + 1, j + 1, k + 1);
	}

	template<typename TDataType, typename Real, typename Coord>
	__global__ void C_UpdateSDF(
		DistanceField3D<TDataType> aDistance,
		DistanceField3D<TDataType> bDistance,
		Coord m_h,
		Coord m_left,
		DArray3D<Real> m_distance,
		int boolType)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		uint nx = m_distance.nx();
		uint ny = m_distance.ny();
		uint nz = m_distance.nz();

		if (i >= nx || j >= ny || k >= nz) return;

		Coord point = m_left + Coord(i * m_h[0], j * m_h[1], k * m_h[2]);

		Real a;
		Coord normal;
		aDistance.getDistance(point, a, normal);

		Real b;
		bDistance.getDistance(point, b, normal);


		switch (boolType)
		{
		case 0://Intersect
			m_distance(i, j, k) = a > b ? a : b;
			break;
		case 1://Union
			m_distance(i, j, k) = a > b ? b : a;
			break;
		case 2://Minus
			m_distance(i, j, k) = a > -b ? a : -b;
			break;
		default:
			break;
		}
	}

	template<typename TDataType>
	VolumeBool<TDataType>::VolumeBool()
	{
		this->inPadding()->setValue(5);
		this->inSpacing()->setValue(0.01);

	}

	template<typename TDataType>
	VolumeBool<TDataType>::~VolumeBool()
	{
	}

	template<typename TDataType>
	void VolumeBool<TDataType>::CalcuSDFGrid(DistanceField3D<TDataType> aDistance,
		DistanceField3D<TDataType> bDistance,
		DistanceField3D<TDataType>& tDistance)
	{
		Vec3f min_box(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
		Vec3f max_box(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());

		min_box = min_box.minimum(aDistance.lowerBound());
		min_box = min_box.minimum(bDistance.lowerBound());

		max_box = max_box.maximum(aDistance.upperBound());
		max_box = max_box.maximum(bDistance.upperBound());

		Real dx = this->inSpacing()->getData();

		int ni = std::floor((max_box[0] - min_box[0]) / dx);
		int nj = std::floor((max_box[1] - min_box[1]) / dx);
		int nk = std::floor((max_box[2] - min_box[2]) / dx);

		tDistance.setSpace(min_box, max_box, ni - 1, nj - 1, nk - 1);

		printf("Uniform boolean grids: %f %f %f, %f %f %f, %f, %d %d %d \n", min_box[0], min_box[1], min_box[2], max_box[0], max_box[1], max_box[2], dx, ni, nj, nk);
	}

	template<typename TDataType>
	void VolumeBool<TDataType>::resetStates()
	{
		std::clock_t Time0 = clock();

		//1、获取数据
		DistanceField3D<TDataType> aDistance = this->inA()->getDataPtr()->getSDF();
		DistanceField3D<TDataType> bDistance = this->inB()->getDataPtr()->getSDF();

		CArray3D<Real> CaDistance;
		CaDistance.assign(aDistance.getMDistance());

		CArray3D<Real> CbDistance;
		CbDistance.assign(bDistance.getMDistance());


		if (this->outSDF()->isEmpty()) {
			this->outSDF()->allocate();
		}
		DistanceField3D<TDataType>& tDistance = this->outSDF()->getDataPtr()->getSDF();

		///2、计算公共区域
		CalcuSDFGrid(aDistance, bDistance, tDistance);

		//3、计算Bool结果
		cuExecute3D(make_uint3(tDistance.getMDistance().nx(), tDistance.getMDistance().ny(), tDistance.getMDistance().nz()),
			C_UpdateSDF,
			aDistance,
			bDistance,
			tDistance.getH(),
			tDistance.lowerBound(),
			tDistance.getMDistance(),
			this->varBoolType()->getDataPtr()->currentKey());

	
		//fast Marching method[@ Implementing and Analysing the fast Marching Method]
		//初始化-构建一个更大的SDF，避免频繁的边界判断
		DArray3D<Real> bigSDFValue(tDistance.nx() + 2, tDistance.ny() + 2, tDistance.nz() + 2);

		DArray<int> pointType;
		pointType.resize(tDistance.nx() * tDistance.ny() * tDistance.nz());//   know = 1; unkonw = 0;

		cuExecute3D(make_uint3(bigSDFValue.nx(), bigSDFValue.ny(), bigSDFValue.nz()),
			C_CopySDFValue,
			bigSDFValue,
			tDistance.getMDistance(),
			pointType,
			aDistance,
			bDistance,
			tDistance.getH(),
			tDistance.lowerBound());


		//fast Marching 核心算法
		int inter = 0;
		int totalVNum = 0;
		Reduction<int> reduce;
		//while(totalVNum < pointType.size()){
		while (totalVNum < pointType.size()||inter<10){
			cuExecute3D(make_uint3(bigSDFValue.nx(), bigSDFValue.ny(), bigSDFValue.nz()),
				C_FastMarching,
				bigSDFValue,
				tDistance.getMDistance(),
				tDistance.getH()[0],
				pointType);			
			totalVNum = reduce.accumulate(pointType.begin(), pointType.size());

			if (totalVNum > (pointType.size() - 1))
				inter++;

			//printf("VolumeBool: fast iteration %d %d \n", inter, totalVNum);
		}

		cuExecute3D(make_uint3(tDistance.nx(), tDistance.ny(), tDistance.nz()),
			C_GetSDFValue,
			tDistance.getMDistance(),
			bigSDFValue);

		this->outSDF()->getDataPtr()->setSDF(tDistance);

		std::clock_t Time1 = clock();
		std::printf("Union boolean Operation time: %d clocks \n", Time1 - Time0);
	}

	DEFINE_CLASS(VolumeBool);
}