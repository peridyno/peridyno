#include "ParticleWriter.h"

#include <sstream>
#include <iostream>
#include <fstream>

namespace dyno
{
	IMPLEMENT_TCLASS(ParticleWriter, TDataType)

	template<typename TDataType>
	ParticleWriter<TDataType>::ParticleWriter()
	: OutputModule()
	{
	
	}

	template<typename TDataType>
	ParticleWriter<TDataType>::~ParticleWriter()
	{
	}


	template<typename TDataType>
	void ParticleWriter<TDataType>::output()
	{
		std::stringstream ss;
		int frameNum = this->getFrameNumber();
		//**********渲染数据需要自动补零
		if (frameNum < 10) ss<<"000";
		else if (frameNum < 100) ss << "00";
		else if (frameNum < 1000) ss << "0";
		//**********
		ss << frameNum;
		

		auto path = this->varOutputPath()->getValue();

		std::string filename = path + ss.str() + std::string(".txt");
		
		auto fileType = this->varFileType()->getValue();
		if (fileType == OpenType::ASCII) 
		{
			OutputASCII(filename);
		}
		else if (fileType == OpenType::binary)
		{
			OutputBinary(filename);
		}
	}

	template<typename TDataType>
	void ParticleWriter<TDataType>::OutputASCII(std::string filename)
	{
		std::fstream output;
		output.open(filename.c_str(), std::ios::out);
		auto& points = this->inPointSet()->getDataPtr()->getPoints();
		int ptNum = points.size();

		output << ptNum << ' ';

		CArray<Coord> hPosition;
		hPosition.resize(ptNum);

		hPosition.assign(points);

		for (int i = 0; i < ptNum; i++) 
		{
			output << hPosition[i][0] << ' ' << hPosition[i][1] << ' ' << hPosition[i][2] << ' ';
		}
		output.close();
	}

	template<typename TDataType>
	void ParticleWriter<TDataType>::OutputBinary(std::string filename)
	{
		std::fstream output;
		output.open(filename.c_str(), std::ios::out | std::ios::binary);

		auto& points = this->inPointSet()->getDataPtr()->getPoints();
		int ptNum = points.size();

		output.write((char*)&ptNum, sizeof(int));

		CArray<Coord> hPosition;
		hPosition.resize(ptNum);
		hPosition.assign(points);

		for (int i = 0; i < ptNum; i++) 
		{
			output.write((char*)&(hPosition[i][0]), sizeof(Real));
			output.write((char*)&(hPosition[i][1]), sizeof(Real));
			output.write((char*)&(hPosition[i][2]), sizeof(Real));
		}

	}



	DEFINE_CLASS(ParticleWriter);
}