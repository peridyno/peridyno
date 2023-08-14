#pragma once
#include "Module/OutputModule.h"
#include "Module/TopologyModule.h"
#include "node.h"
#include "Topology/TriangleSet.h"

#include <string>
#include <memory>

namespace dyno
{
	template <typename TDataType> class TriangleSet;

	template<typename TDataType>
	class PlyExporter : public Node
	{
		DECLARE_TCLASS(PlyExporter, TDataType)
	public:

		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename Quat<Real> TQuat;

		PlyExporter();

		//void update() override;

	public:

		DEF_VAR(std::string, OutputPath, "D:/plyout_", "OutputPath");

		DEF_VAR(unsigned, FrameStep, 1, "FrameStep");

		DEF_VAR(bool, ReCount, false, "ReCount");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "TriangleSet")

		DEF_ARRAY_IN(Vec3f, Vec3f, DeviceType::GPU, "");

		DEF_ARRAY_IN(Matrix, Matrix1, DeviceType::GPU, "");

		DEF_ARRAY_IN(Matrix, Matrix2, DeviceType::GPU, "");


	protected:
		void resetStates() override;

		void updateStates() override;
		

	private:
		std::string file_postfix = ".ply";

	};
}