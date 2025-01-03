#pragma once
#include "Node.h"

#include "Module/OutputModule.h"
#include "Module/TopologyModule.h"

#include "Topology/TriangleSet.h"
#include "TriangleMeshWriter.h"
#include "Topology/PolygonSet.h"

#include <string>
#include <memory>

namespace dyno
{
	template<typename TDataType>
	class ObjExporter : public Node
	{
		DECLARE_TCLASS(ObjExporter, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename Quat<Real> TQuat;

		ObjExporter();

		std::string getNodeType() override { return "IO"; }

		DECLARE_ENUM(OutputType,
			Mesh = 0,
			PointCloud = 1);

	public:
		DEF_VAR(std::string, OutputPath, "D:/File_", "OutputPath");
		//DEF_VAR(std::string, Filename, "", "Filename");
		DEF_VAR(unsigned, StartFrame, 1, "StartFrame");
		DEF_VAR(unsigned, EndFrame, 500000, "EndFrame");
		DEF_VAR(unsigned, FrameStep, 1, "FrameStep");
		DEF_ENUM(OutputType, OutputType, Mesh, "OutputType");

		DEF_INSTANCE_IN(PolygonSet<TDataType>, PolygonSet, "PolygonSet");
		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "TriangleSet");

	protected:
		void resetStates() override;
		void updateStates() override;

		void outputTriangleMesh(std::shared_ptr<TriangleSet<TDataType>> triangleset);

		void outputPolygonSet(std::shared_ptr<PolygonSet<TDataType>> polygonSet);

	private:
		std::string file_postfix = ".obj";
	};
}