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

	}

	template<typename TDataType>
	TriangleMeshWriter<TDataType>::~TriangleMeshWriter()
	{
	}

	template<typename TDataType>
	void TriangleMeshWriter<TDataType>::setNamePrefix(std::string prefix)
	{
		this->name_prefix = prefix;
	}

	template<typename TDataType>
	void TriangleMeshWriter<TDataType>::setOutputPath(std::string path)
	{
		this->output_path = path;
	}

	template<typename TDataType>
	bool TriangleMeshWriter<TDataType>::updatePtr()
	{
		if (this->ptr_TriangleSet == nullptr) {
			return false;
		}
		this->ptr_triangles = &(this->ptr_TriangleSet->getTriangles());
		this->ptr_vertices = &(this->ptr_TriangleSet->getPoints());
	}

	template<typename TDataType>
	bool TriangleMeshWriter<TDataType>::updatePtr(TriangleSet<TDataType> triangle_set)
	{
		if (&triangle_set == nullptr) {
			return false;
		}
		this->ptr_triangles = &(triangle_set.getTriangles());
		this->ptr_vertices = &(triangle_set.getPoints());
	}


	template<typename TDataType>
	bool TriangleMeshWriter<TDataType>::outputSurfaceMesh()
	{
		if (time_idx % 3 != 0)
		{
			time_idx++;
			return true;
		}
		time_idx++;

		if (this->ptr_vertices == nullptr || this->ptr_triangles == nullptr) {
			printf("------Triangle Mesh Writer: array nullptr \n");
			return false;
		}

		std::stringstream ss; ss << m_output_index;
		std::string filename = output_path + "/" + this->name_prefix + ss.str() + this->file_postfix;
		std::ofstream output(filename.c_str(), std::ios::out);

		if (!output.is_open()) {
			printf("------Triangle Mesh Writer: open file failed \n");
			return false;
		}

		std::cout <<  "------Triangle Mesh Writer Action!------ "<< std::endl;

		// update pointer, 
		this->updatePtr();

		CArray<Coord> host_vertices;
		CArray<Triangle> host_triangles;

		host_vertices.resize((*(this->ptr_vertices)).size());
		host_triangles.resize((*(this->ptr_triangles)).size());

		host_vertices.assign(*(this->ptr_vertices));
		host_triangles.assign(*(this->ptr_triangles));

		for (uint i = 0; i < host_vertices.size(); ++i) {
			output << "v " << host_vertices[i][0] << " " << host_vertices[i][1] << " " << host_vertices[i][2] << std::endl;
		}
		for (uint i = 0; i < host_triangles.size(); ++i) {
			output << "f " << host_triangles[i][0] + 1 << " " << host_triangles[i][1] + 1 << " " << host_triangles[i][2] + 1 << std::endl;
		}

		host_vertices.clear();
		host_triangles.clear();

		this->m_output_index++;
		return true;
	}

	template<typename TDataType>
	bool TriangleMeshWriter<TDataType>::outputSurfaceMesh(TriangleSet<TDataType> triangleset)
	{	
		auto frame_step = this->varFramStep()->getData();
		auto current_frame = this->inFrameNumber()->getData();
		auto out_number = this->inFrameNumber()->getData();

		if (current_frame % frame_step != 0)
		{
			printf("Ìø¹ý¸ÃÖ¡\n");
			return false;
		}

		if (frame_step > 1)
		{
			out_number = floor(current_frame / frame_step);
		}


		std::stringstream ss; ss << out_number;
		std::string filename = this->varOutputPath()->getData() + "/" + this->varPrefix()->getData() + ss.str() + this->file_postfix;
		std::ofstream output(filename.c_str(), std::ios::out);
		
		std::cout << filename << std::endl;

		if (!output.is_open()) {
			printf("------Triangle Mesh Writer: open file failed \n");
			return false;
		}

		std::cout << "------Triangle Mesh Writer Action!------ " << std::endl;

		// update pointer, 
		this->updatePtr(triangleset);

		CArray<Coord> host_vertices;
		CArray<Triangle> host_triangles;

		host_vertices.resize((*(this->ptr_vertices)).size());
		host_triangles.resize((*(this->ptr_triangles)).size());

		host_vertices.assign(*(this->ptr_vertices));
		host_triangles.assign(*(this->ptr_triangles));

		for (uint i = 0; i < host_vertices.size(); ++i) {
			output << "v " << host_vertices[i][0] << " " << host_vertices[i][1] << " " << host_vertices[i][2] << std::endl;
		}
		for (uint i = 0; i < host_triangles.size(); ++i) {
			output << "f " << host_triangles[i][0] + 1 << " " << host_triangles[i][1] + 1 << " " << host_triangles[i][2] + 1 << std::endl;
		}

		host_vertices.clear();
		host_triangles.clear();
		
		this->m_output_index++;
		return true;
	}

	template<typename TDataType>
	void TriangleMeshWriter<TDataType>::updateImpl()
	{
		//printf("===========Triangle Mesh Writer============\n");

		//if (this->m_output_index >= this->max_output_files) { return; }

		//if (this->current_idle_frame <= 0) {
			//this->current_idle_frame = this->idle_frame_num;

		this->outputSurfaceMesh();

		//}
		//else {
		//	this->current_idle_frame--;
		//}
		
		if (this->inFrameNumber()->getData() > this->varEnd()->getData()) { return; }
		else if (this->inFrameNumber()->getData() < this->varStart()->getData()) { return; }
		else 
		{
			this->outputSurfaceMesh(this->inTriangleSet()->getData());
		}


	}

	DEFINE_CLASS(TriangleMeshWriter);
}