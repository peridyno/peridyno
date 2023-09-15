#include "TriangleMeshWriter.h"
#include "Module/OutputModule.h"

#include <sstream>
#include <iostream>
#include <fstream>

namespace dyno
{
	IMPLEMENT_TCLASS(TriangleMeshWriter, TDataType)

	template<typename TDataType>
	TriangleMeshWriter<TDataType>::TriangleMeshWriter() : OutputModule()
	{
		this->varEnd()->setRange(1, 100000);
		
		this->inFrameNumber()->tagOptional(true);
	}

	template<typename TDataType>
	TriangleMeshWriter<TDataType>::~TriangleMeshWriter()
	{
	}



	template<typename TDataType>
	void TriangleMeshWriter<TDataType>::output()
	{
		auto mode = this->varOutputType()->getValue();

		if (mode == OutputType::TriangleMesh) 
		{
			auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->inTopology()->getDataPtr());
			outputSurfaceMesh(triSet);
		}
		else if (mode == OutputType::PointCloud)
		{
			auto ptSet = TypeInfo::cast<PointSet<TDataType>>(this->inTopology()->getDataPtr());
			outputPointCloud(ptSet);
		}

	}

	template<typename TDataType>
	void TriangleMeshWriter<TDataType>::outputSurfaceMesh(std::shared_ptr<TriangleSet<TDataType>> triangleset)
	{
		int out_number = getFrameNumber();

		std::stringstream ss; ss << out_number;
		std::string filename = this->varOutputPath()->getData() + ss.str() + this->file_postfix;// 
		std::ofstream output(filename.c_str(), std::ios::out);

		std::cout << filename << std::endl;

		if (!output.is_open()) {
			printf("------Triangle Mesh Writer: open file failed \n");
			return;
		}

		std::cout << "------Triangle Mesh Writer Action!------ " << std::endl;


		CArray<Coord> host_vertices;
		CArray<TopologyModule::Triangle> host_triangles;


		if (triangleset->getPoints().size())
		{
			host_vertices.assign(triangleset->getPoints());
		}
		if (triangleset->getTriangles().size())
		{
			host_triangles.assign(triangleset->getTriangles());
		}


		for (uint i = 0; i < host_vertices.size(); ++i) {
			output << "v " << host_vertices[i][0] << " " << host_vertices[i][1] << " " << host_vertices[i][2] << std::endl;
		}
		for (uint i = 0; i < host_triangles.size(); ++i) {
			output << "f " << host_triangles[i][0] + 1 << " " << host_triangles[i][1] + 1 << " " << host_triangles[i][2] + 1 << std::endl;
		}
		output.close();


		host_vertices.clear();
		host_triangles.clear();


		return;
	}

	template<typename TDataType>
	void TriangleMeshWriter<TDataType>::outputPointCloud(std::shared_ptr<PointSet<TDataType>> pointset)
	{
		int out_number = getFrameNumber();

		std::stringstream ss; ss << out_number;
		std::string filename = this->varOutputPath()->getData() + ss.str() + this->file_postfix;// 
		std::ofstream output(filename.c_str(), std::ios::out);

		std::cout << filename << std::endl;

		if (!output.is_open()) {
			printf("------Triangle Mesh Writer: open file failed \n");
			return;
		}

		std::cout << "------Pointcloud Writer Action!------ " << std::endl;

		CArray<Coord> host_vertices;

		if (pointset->getPoints().size())
		{
			host_vertices.assign(pointset->getPoints());
		}

		for (uint i = 0; i < host_vertices.size(); ++i) {
			output << "v " << host_vertices[i][0] << " " << host_vertices[i][1] << " " << host_vertices[i][2] << std::endl;
		}

		output.close();

		host_vertices.clear();


		return;
	}




	DEFINE_CLASS(TriangleMeshWriter);
}