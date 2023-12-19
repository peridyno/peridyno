#include "EigenValueWriter.h"

#include <sstream>
#include <iostream>
#include <fstream>
#include <stdlib.h>

namespace dyno
{
	IMPLEMENT_TCLASS(EigenValueWriter, TDataType)

	template<typename TDataType>
	EigenValueWriter<TDataType>::EigenValueWriter()
	: OutputModule()
	{
	}

	template<typename TDataType>
	EigenValueWriter<TDataType>::~EigenValueWriter()
	{
	}
	

	template<typename TDataType>
	void EigenValueWriter<TDataType>::output()
	{
		auto path = this->varOutputPath()->getValue();

		std::stringstream ss; ss << getFrameNumber();
		std::string filename = path + ss.str()  + std::string(".txt");
		std::fstream output(filename.c_str(), std::ios::out);


		int pNum = this->inTransform()->size();
		output << pNum << ' ';

		/*CArray<Coord> hPosition;
		hPosition.resize(pNum);
		hPosition.assign(this->inPosition()->getData());*/

		CArray <Transform3f> tm;
		tm.resize(pNum);
		tm.assign(this->inTransform()->getData());

		for (int i = 0; i < pNum; i++)
		{
			Vec3f dM = tm[i].scale();
			Mat3f rM = tm[i].rotation();
			Coord pos = tm[i].translation();
			pos[1] += 21.37;

			//Particle Position
			char buf1[50], buf2[50], buf3[50];
			sprintf(buf1, "%f", (double)pos[0]);
			sprintf(buf2, "%f", (double)pos[1]);
			sprintf(buf3, "%f", (double)pos[2]);
			output << buf1 << ' ' << buf2 << ' ' << buf3 << ' ';

			//Eigen Values
			char d1[50], d2[50], d3[50];
			sprintf(d1, "%f", (double)dM[0]);
			sprintf(d2, "%f", (double)dM[1]);
			sprintf(d3, "%f", (double)dM[2]);
			output << d1 << ' ' << d2 << ' ' << d3 << ' ';

			//Rotation Matrix
			//v1
			char v11[50], v12[50], v13[50];
			sprintf(v11, "%f", (double)rM(0, 0));
			sprintf(v12, "%f", (double)rM(1, 0));
			sprintf(v13, "%f", (double)rM(2, 0));
			output << v11 << ' ' << v12 << ' ' << v13 << ' ';

			//v2
			char v21[50], v22[50], v23[50];
			sprintf(v21, "%f", (double)rM(0, 1));
			sprintf(v22, "%f", (double)rM(1, 1));
			sprintf(v23, "%f", (double)rM(2, 1));
			output << v21 << ' ' << v22 << ' ' << v23 << ' ';

			//v3
			char v31[50], v32[50], v33[50];
			sprintf(v31, "%f", (double)rM(0, 2));
			sprintf(v32, "%f", (double)rM(1, 2));
			sprintf(v33, "%f", (double)rM(2, 2));
			output << v31 << ' ' << v32 << ' ' << v33 << ' ';
		}

		output.close();

	}

	DEFINE_CLASS(EigenValueWriter);
}