/*
This Module is designed to output mesh file of TriangleSet;
the output file format: obj
*/

#pragma once
#include "Module/OutputModule.h"
#include "Module/TopologyModule.h"

#include "Topology/TriangleSet.h"

#include <string>
#include <memory>

namespace dyno
{
	template<typename TDataType>
	class TriangleMeshWriter : public OutputModule
	{
		DECLARE_TCLASS(TriangleMeshWriter, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;


		DECLARE_ENUM(OutputType,
			TriangleMesh = 0,
			PointCloud = 1);

		TriangleMeshWriter();
		virtual ~TriangleMeshWriter();

		void outputSurfaceMesh(std::shared_ptr<TriangleSet<TDataType>> triangleset);
		void outputPointCloud(std::shared_ptr<PointSet<TDataType>> pointset);

		void output()override;


	public:

		DEF_INSTANCE_IN(TopologyModule, Topology, "Input TriangleSet");
		DEF_ENUM(OutputType,OutputType,OutputType::TriangleMesh,"OutputType")




	protected:

		std::string file_postfix = ".obj";
		int mFileIndex = 0;
		int count = -1;
		bool skipFrame = false;

	};
}