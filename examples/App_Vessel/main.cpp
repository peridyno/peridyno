#include <string>
#include "simple.h"
#include "simple.cpp"
#include "DistanceField3D.h"
#include "DistanceField3D.cpp"

main()
{
	cout << "start the nornal test**************************************************************" << endl;
	float timestep = 0.001;
	int frame = 0;
	Simple simple;
	std::string outpath("F:/Github_rep/Data/");

	std::string fill_name("NORMAL1");
	std::stringstream strout1, strout2, strout3, strout4, strout5, strout01, strout02, strout03, strout04, strout11, strout12, strout13, strout14, strout21, strout22, strout23, strout24;
	strout1 << "../../model/" << fill_name << "/pv1.sdf";
	strout2 << "../../model/" << fill_name << "/inlet1.sdf";
	strout3 << "../../model/" << fill_name << "/inlet1.sdf";
	strout4 << "../../model/" << fill_name << "/outlet11.sdf";
	strout5 << "../../model/" << fill_name << "/outlet21.sdf";

	strout01 << "../../model/" << fill_name << "/face1.obj";
	strout02 << "../../model/" << fill_name << "/face2.obj";
	strout03 << "../../model/" << fill_name << "/face3.obj";
	strout04 << "../../model/" << fill_name << "/face4.obj";
	strout11 << "../../model/" << fill_name << "/face11.obj";
	strout12 << "../../model/" << fill_name << "/face12.obj";
	strout13 << "../../model/" << fill_name << "/face13.obj";
	strout14 << "../../model/" << fill_name << "/face14.obj";
	strout21 << "../../model/" << fill_name << "/face21.obj";
	strout22 << "../../model/" << fill_name << "/face22.obj";
	strout23 << "../../model/" << fill_name << "/face23.obj";
	strout24 << "../../model/" << fill_name << "/face24.obj";

	std::string pv = strout1.str();
	std::string in1 = strout2.str();
	std::string in2 = strout3.str();
	std::string out1 = strout4.str();
	std::string out2 = strout5.str();

	std::string f1 = strout01.str();
	std::string f2 = strout02.str();
	std::string f3 = strout03.str();
	std::string f4 = strout04.str();
	std::string f11 = strout11.str();
	std::string f12 = strout12.str();
	std::string f13 = strout13.str();
	std::string f14 = strout14.str();
	std::string f21 = strout21.str();
	std::string f22 = strout22.str();
	std::string f23 = strout23.str();
	std::string f24 = strout24.str();


	simple.initialize(1, pv, in1, in2, out1, out2, 0, 0);
	std::printf("Initializing liquid\n");

	for (frame = 0; frame < 1; frame++)
	{
		std::printf("--------------------\nFrame %d\n", frame);
		printf("Simulating liquid\n");
		simple.advance(f24, outpath, frame, timestep, 0, 0);
	}
	//simple.export_status(outpath, 0);
	simple.export_velocity_vtk(outpath, 0);
	simple.export_pressure_vtk(outpath, 0);

	//std::string datap("F:/MICCAI_prepare/pv_data/sheet7/NORMAL41_BLOCK_ONE.txt");
	//bool succeded = simple.import_status(datap);
	//if (!succeded)
	//{
	//	printf("Load status failed!");
	//	exit(0);
	//	return;
	//}

	//statistics pressure
	double p_f1 = 0, p_f2 = 0, p_f3 = 0, p_f4 = 0, p_f11 = 0, p_f12 = 0, p_f13 = 0, p_f14 = 0, p_f21 = 0, p_f22 = 0, p_f23 = 0, p_f24 = 0;
	p_f1 = simple.statistics_pressure(f1);
	p_f2 = simple.statistics_pressure(f2);
	p_f3 = simple.statistics_pressure(f3);
	p_f4 = simple.statistics_pressure(f4);
	p_f11 = simple.statistics_pressure(f11);
	p_f12 = simple.statistics_pressure(f12);
	p_f13 = simple.statistics_pressure(f13);
	p_f14 = simple.statistics_pressure(f14);
	p_f21 = simple.statistics_pressure(f21);
	p_f22 = simple.statistics_pressure(f22);
	p_f23 = simple.statistics_pressure(f23);
	p_f24 = simple.statistics_pressure(f24);

	std::cout << "the average pressure of face1 and face2 is: " << p_f1 << " " << p_f2 << " " << p_f1 - p_f2 << std::endl;
	std::cout << "the average pressure of face2 and face3 is: " << p_f2 << " " << p_f3 << " " << p_f2 - p_f3 << std::endl;
	std::cout << "the average pressure of face3 and face4 is: " << p_f3 << " " << p_f4 << " " << p_f3 - p_f4 << std::endl;
	std::cout << "the average pressure of face11 and face12 is: " << p_f11 << " " << p_f12 << " " << p_f11 - p_f12 << std::endl;
	std::cout << "the average pressure of face12 and face13 is: " << p_f12 << " " << p_f13 << " " << p_f12 - p_f13 << std::endl;
	std::cout << "the average pressure of face13 and face14 is: " << p_f13 << " " << p_f14 << " " << p_f13 - p_f14 << std::endl;
	std::cout << "the average pressure of face21 and face22 is: " << p_f21 << " " << p_f22 << " " << p_f21 - p_f22 << std::endl;
	std::cout << "the average pressure of face22 and face23 is: " << p_f22 << " " << p_f23 << " " << p_f22 - p_f23 << std::endl;
	std::cout << "the average pressure of face23 and face24 is: " << p_f23 << " " << p_f24 << " " << p_f23 - p_f24 << std::endl;

	std::cout << std::endl << std::endl;
}


