#include "PointSet.h"
#include <fstream>
#include <iostream>
#include <sstream>

namespace dyno
{
	IMPLEMENT_TCLASS(PointSet, TDataType)

	template<typename TDataType>
	PointSet<TDataType>::PointSet()
		: TopologyModule()
	{
		//this->setUpdateAlways(true);
	}

	template<typename TDataType>
	PointSet<TDataType>::~PointSet()
	{
		mCoords.clear();
	}

	template<typename TDataType>
	void PointSet<TDataType>::loadObjFile(std::string filename)
	{
		if (filename.size() < 5 || filename.substr(filename.size() - 4) != std::string(".obj")) {
			std::cerr << "Error: Expected OBJ file with filename of the form <name>.obj.\n";
			return;
		}

		std::ifstream infile(filename);
		if (!infile) {
			std::cerr << "Failed to open. Terminating.\n";
			return;
		}

		int ignored_lines = 0;
		std::string line;
		std::vector<Coord> vertList;
		while (!infile.eof()) {
			std::getline(infile, line);

			//.obj files sometimes contain vertex normals indicated by "vn"
			if (line.substr(0, 1) == std::string("v") && line.substr(0, 2) != std::string("vn")) {
				std::stringstream data(line);
				char c;
				Coord point;
				data >> c >> point[0] >> point[1] >> point[2];
				vertList.push_back(point);
			}
			else {
				++ignored_lines;
			}
		}
		infile.close();

		std::cout << "Total number of particles: " << vertList.size() << std::endl;

		setPoints(vertList);

		vertList.clear();
	}

	template<typename TDataType>
	void PointSet<TDataType>::copyFrom(PointSet<TDataType>& pointSet)
	{
		mCoords.assign(pointSet.getPoints());
	}

	template<typename TDataType>
	void PointSet<TDataType>::setPoints(std::vector<Coord>& pos)
	{
		mCoords.resize(pos.size());
		mCoords.assign(pos);

		tagAsChanged();
	}

	template<typename TDataType>
	void PointSet<TDataType>::setPoints(DArray<Coord>& pos)
	{
		mCoords.resize(pos.size());
		mCoords.assign(pos);

		tagAsChanged();
	}

	template<typename TDataType>
	void PointSet<TDataType>::setSize(int size)
	{
		mCoords.resize(size);
		mCoords.reset();
	}

	template<typename TDataType>
	void PointSet<TDataType>::requestBoundingBox(Coord& lo, Coord& hi)
	{
		Reduction<Coord> reduce;
		lo = reduce.minimum(mCoords.begin(), mCoords.size());
		hi = reduce.maximum(mCoords.begin(), mCoords.size());
	}

	template <typename Real, typename Coord>
	__global__ void PS_Scale(
		DArray<Coord> vertex,
		Real s)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vertex.size()) return;
		//return;
		vertex[pId] = vertex[pId] * s;
	}

	template<typename TDataType>
	void PointSet<TDataType>::scale(const Real s)
	{
		cuExecute(mCoords.size(), PS_Scale, mCoords, s);
	}

	template <typename Coord>
	__global__ void PS_Scale(
		DArray<Coord> vertex,
		Coord s)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vertex.size()) return;

		Coord pos_i = vertex[pId];
		vertex[pId] = Coord(pos_i[0] * s[0], pos_i[1] * s[1], pos_i[2] * s[2]);
	}

	template<typename TDataType>
	void PointSet<TDataType>::scale(const Coord s)
	{
		cuExecute(mCoords.size(), PS_Scale, mCoords, s);
	}

	template <typename Coord>
	__global__ void PS_Translate(
		DArray<Coord> vertex,
		Coord t)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vertex.size()) return;

		vertex[pId] = vertex[pId] + t;
	}


	template<typename TDataType>
	void PointSet<TDataType>::translate(const Coord t)
	{
		cuExecute(mCoords.size(), PS_Translate, mCoords, t);
	}

	template <typename Coord>
	__global__ void PS_Rotate(
		DArray<Coord> vertex,
		Coord theta,
		Coord origin
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vertex.size()) return;
		//return;
		Coord pos = vertex[pId];

		Real x = pos[0];
		Real y = pos[1];
		Real z = pos[2];

		pos[1] = y * cos(theta[1]) - z * sin(theta[1]);
		pos[2] = y * sin(theta[1]) + z * cos(theta[1]);

		x = pos[0];
		y = pos[1];
		z = pos[2];

		pos[0] = x * cos(theta[0]) - y * sin(theta[0]);
		pos[1] = x * sin(theta[0]) + y * cos(theta[0]);

		x = pos[0];
		y = pos[1];
		z = pos[2];

		pos[2] = z * cos(theta[2]) - x * sin(theta[2]);
		pos[0] = z * sin(theta[2]) + x * cos(theta[2]);

		vertex[pId] = pos;
	}


	template<typename TDataType>
	void PointSet<TDataType>::rotate(const Coord angle)
	{
		cuExecute(mCoords.size(), PS_Rotate, mCoords, angle, Coord(0.0f));
	}

	template <typename Coord>
	__global__ void PS_Rotate(
		DArray<Coord> vertex,
		Quat<Real> q)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vertex.size()) return;
		Mat3f rot = q.toMatrix3x3();

		vertex[pId] = rot * vertex[pId];
	}

	template<typename TDataType>
	void PointSet<TDataType>::rotate(const Quat<Real> q)
	{
		cuExecute(mCoords.size(), PS_Rotate, mCoords, q);
	}

	template<typename TDataType>
	bool PointSet<TDataType>::isEmpty()
	{
		return mCoords.size() == 0;
	}

	template<typename TDataType>
	void PointSet<TDataType>::clear()
	{
		mCoords.clear();
	}

	DEFINE_CLASS(PointSet);
}