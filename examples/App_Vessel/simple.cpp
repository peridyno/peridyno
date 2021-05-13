#include "simple.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <random>
#include<Eigen/Sparse>
#include<Eigen/IterativeLinearSolvers>

#include"vtkSmartPointer.h"
#include"vtkDoubleArray.h"
#include"vtkPolyData.h"
#include"vtkPoints.h"
#include"vtkPointData.h"
#include"vtkCellArray.h"
#include"vtkPolyDataWriter.h"

#include "array3_utils.h"
#include "levelset_util.h"

//#define BLOCK

void Simple::initialize(double dx_,std::string fluid, std::string inlet1, std::string inlet2, std::string outlet1, std::string outlet2,double alpha,double theta)
{
	//pipe
	//inlet_direction1 = Vec3d(-cos(alpha)*cos(theta), cos(alpha)*sin(theta), -sin(alpha));
	//inlet_direction2 = Vec3d(-cos(alpha)*cos(theta), cos(alpha)*sin(theta), -sin(alpha));
	////YU_JUAN
	//inlet_direction1 = Vec3d(0.333928,-0.575716,-0.746353);
	//inlet_direction2 = Vec3d(0.333928, -0.575716, -0.746353);
	//YANG_ZHONGYUN
	//inlet_direction1 = Vec3d(0.121644, -0.481770, -0.867813);
	//inlet_direction2 = Vec3d(0.121644, -0.481770, -0.867813);
	//YANG_ZHONGYUN
	//inlet_direction1 = Vec3d(0.566940, -0.393927, -0.723464);
	//inlet_direction2 = Vec3d(0.566940, -0.393927, -0.723464);
	//BAI_YUN
	//inlet_direction1 = Vec3d(0.302251, -0.769818, -0.562161);
	//inlet_direction2 = Vec3d(0.302251, -0.769818, -0.562161);
	//NORMAL
	inlet_direction1 = Vec3d(0.490999, -0.356215, -0.795004);
	inlet_direction2 = Vec3d(0.490999, -0.356215, -0.795004);
	//inlet_direction1 = Vec3d(0.213074, -0.507018, -0.835184);
	//inlet_direction2 = Vec3d(0.213074, -0.507018, -0.835184);
	//inlet_direction1 = Vec3d(0.176353, -0.527966, -0.830754);
	//inlet_direction2 = Vec3d(0.176353, -0.527966, -0.830754);
	//NORMAL2
	//inlet_direction1 = Vec3d(0.457737, -0.447532, -0.768240);
	//inlet_direction2 = Vec3d(0.457737, -0.447532, -0.768240);
	//NORMAL3
	//inlet_direction1 = Vec3d(0.482327, -0.326915, -0.812704);
	//inlet_direction2 = Vec3d(0.482327, -0.326915, -0.812704);
	//NORMAL4
	//inlet_direction1 = Vec3d(0.620663, -0.626506, -0.471452);
	//inlet_direction2 = Vec3d(0.620663, -0.626506, -0.471452);
	//inlet_direction1 = Vec3d(0.612902, -0.632944, -0.473005);
	//inlet_direction2 = Vec3d(0.612902, -0.632944, -0.473005);
	//inlet_direction1 = Vec3d(0.676093, -0.563848, -0.474314);
	//inlet_direction2 = Vec3d(0.676093, -0.563848, -0.474314);
	//inlet_direction1 = Vec3d(0.584255, -0.692331, -0.423467);
	//inlet_direction2 = Vec3d(0.584255, -0.692331, -0.423467);
	//inlet_magnitude1 = 0.2;
	//inlet_magnitude2 = 0.2;
	inlet_magnitude1 = 0.1364;
	inlet_magnitude2 = 0.1364;

	mfd::DistanceField3D fluidSDF(fluid);
	mfd::DistanceField3D inletSDF1(inlet1);
	mfd::DistanceField3D inletSDF2(inlet2);
	mfd::DistanceField3D outletSDF1(outlet1);
	mfd::DistanceField3D outletSDF2(outlet2);

	//Grid dimensions
	lower_vertex = fluidSDF.p0;
	Vec3d upper_vertex = fluidSDF.p1;
	cout << "the lower vertex and upper vertex is: " << lower_vertex << " " << upper_vertex << endl;
	dx_mm =dx_;
	dx =dx_*0.001;
	ni = floor((upper_vertex[0]-lower_vertex[0])/dx_mm);
	nj = floor((upper_vertex[1] - lower_vertex[1]) / dx_mm);
	nk = floor((upper_vertex[2] - lower_vertex[2]) / dx_mm);
	std::cout << ni << " " << nj << " " << nk << std::endl;

	u.resize(ni + 1, nj, nk); u_starstar.resize(ni + 1, nj, nk); u_prime.resize(ni + 1, nj, nk); u_init.resize(ni + 1, nj, nk);
	v.resize(ni, nj + 1, nk); v_starstar.resize(ni, nj + 1, nk); v_prime.resize(ni, nj + 1, nk); v_init.resize(ni, nj + 1, nk);
	w.resize(ni, nj, nk + 1); w_starstar.resize(ni, nj, nk + 1); w_prime.resize(ni, nj, nk + 1); w_init.resize(ni, nj, nk + 1);
	u.set_zero(); u_starstar.set_zero(); u_prime.set_zero(); u_init.set_zero();
	v.set_zero(); v_starstar.set_zero(); v_prime.set_zero(); v_init.set_zero();
	w.set_zero(); w_starstar.set_zero(); w_prime.set_zero(); w_init.set_zero();

	p.resize(ni, nj, nk); p_star.resize(ni, nj, nk); p_prime.resize(ni, nj, nk);
	p.set_zero(); p_star.set_zero(); p_prime.set_zero();

	//initialize the signed distance field for liquid, inlet and outlet.
	initialize_sdf(fluidSDF, inletSDF1, inletSDF2, outletSDF1, outletSDF2, liquid_phi, u_liquid_phi, v_liquid_phi, w_liquid_phi, p_identifer, u_identifer, v_identifer, w_identifer);

	std::cout << "initialize successful" << endl;
}

void Simple::initialize(Vec3d pos, double dx_, int ni_, int nj_, int nk_, mfd::DistanceField3D& fluidSDF, mfd::DistanceField3D& inletSDF, mfd::DistanceField3D& outletSDF,double alpha)
{
	ni = ni_; nj = nj_; nk = nk_;

	lower_vertex = pos;

	dx_mm = dx_;
	dx = dx_ * 0.001;
	std::cout << ni << " " << nj << " " << nk << std::endl;
	u.resize(ni + 1, nj, nk); u_starstar.resize(ni + 1, nj, nk); u_prime.resize(ni + 1, nj, nk); u_init.resize(ni + 1, nj, nk);
	v.resize(ni, nj + 1, nk); v_starstar.resize(ni, nj + 1, nk); v_prime.resize(ni, nj + 1, nk); v_init.resize(ni, nj + 1, nk);
	w.resize(ni, nj, nk + 1); w_starstar.resize(ni, nj, nk + 1); w_prime.resize(ni, nj, nk + 1); w_init.resize(ni, nj, nk + 1);
	u.set_zero(); u_starstar.set_zero(); u_prime.set_zero(); u_init.set_zero();
	v.set_zero(); v_starstar.set_zero(); v_prime.set_zero(); v_init.set_zero();
	w.set_zero(); w_starstar.set_zero(); w_prime.set_zero(); w_init.set_zero();

	p.resize(ni, nj, nk); p_star.resize(ni, nj, nk); p_prime.resize(ni, nj, nk);
	p.set_zero(); p_star.set_zero(); p_prime.set_zero();

	//initialize the signed distance field and label for pressure and velocity
	if (alpha == 0.0)
	{
		initialize_sdf(fluidSDF, inletSDF, inletSDF, outletSDF, outletSDF, liquid_phi, u_liquid_phi, v_liquid_phi, w_liquid_phi, p_identifer, u_identifer, v_identifer, w_identifer);
		initialize_sdf(fluidSDF, inletSDF, inletSDF, outletSDF, outletSDF, liquid_phi_init, u_liquid_phi_init, v_liquid_phi_init, w_liquid_phi_init, p_identifer_init, u_identifer_init, v_identifer_init, w_identifer_init);
	}
	else
	{
		initialize_sdf(fluidSDF, inletSDF, inletSDF, outletSDF, outletSDF, liquid_phi_init, u_liquid_phi_init, v_liquid_phi_init, w_liquid_phi_init, p_identifer_init, u_identifer_init, v_identifer_init, w_identifer_init);

		fluidSDF.RotationDistance(alpha);
		inletSDF.RotationDistance(alpha);
		outletSDF.RotationDistance(alpha);

		initialize_sdf(fluidSDF, inletSDF, inletSDF, outletSDF, outletSDF, liquid_phi, u_liquid_phi, v_liquid_phi, w_liquid_phi, p_identifer, u_identifer, v_identifer, w_identifer);
	}
}

void Simple::initialize(double dx_,std::string in_model,std::string in_centerline)
{
	inlet_magnitude1 = 0.1364;
	inlet_magnitude2 = 0.1364;
	cout << "it is ok here" << endl;
	mfd::DistanceField3D fluidSDF(in_model);
	cout << "coupute sdf finished" << endl;

	//Grid dimensions
	lower_vertex = fluidSDF.p0;
	Vec3d upper_vertex = fluidSDF.p1;
	cout << "the lower vertex and upper vertex is: " << lower_vertex << " " << upper_vertex << endl;
	dx_mm = dx_;
	dx = dx_ * 0.001;
	ni = floor((upper_vertex[0] - lower_vertex[0]) / dx_mm);
	nj = floor((upper_vertex[1] - lower_vertex[1]) / dx_mm);
	nk = floor((upper_vertex[2] - lower_vertex[2]) / dx_mm);
	std::cout << ni << " " << nj << " " << nk << std::endl;

	u.resize(ni + 1, nj, nk); u_starstar.resize(ni + 1, nj, nk); u_prime.resize(ni + 1, nj, nk); u_init.resize(ni + 1, nj, nk);
	v.resize(ni, nj + 1, nk); v_starstar.resize(ni, nj + 1, nk); v_prime.resize(ni, nj + 1, nk); v_init.resize(ni, nj + 1, nk);
	w.resize(ni, nj, nk + 1); w_starstar.resize(ni, nj, nk + 1); w_prime.resize(ni, nj, nk + 1); w_init.resize(ni, nj, nk + 1);
	u.set_zero(); u_starstar.set_zero(); u_prime.set_zero(); u_init.set_zero();
	v.set_zero(); v_starstar.set_zero(); v_prime.set_zero(); v_init.set_zero();
	w.set_zero(); w_starstar.set_zero(); w_prime.set_zero(); w_init.set_zero();

	p.resize(ni, nj, nk); p_star.resize(ni, nj, nk); p_prime.resize(ni, nj, nk);
	p.set_zero(); p_star.set_zero(); p_prime.set_zero();

	//initialize the signed distance field for liquid, inlet and outlet.
	centerline(in_centerline);
	std::cout << "it is ok here now" << std::endl;
	initialize_sdf(fluidSDF);

	std::cout << "initialize successful" << endl;
}

bool Simple::centerline(std::string in_centerline)
{
	std::cout << "now it is: " << in_centerline << std::endl;
	std::ifstream status_in(in_centerline);
	if (!status_in.good())
	{
		printf("Failed to open files!\n");
		return false;
	}
	//read the points' numbers and lines' numbers;
	int points_numb, lines_numb, faces_numb;
	status_in >> points_numb >> lines_numb >> faces_numb;
	std::cout << "the number of points and lines is: " << points_numb << " " << lines_numb << std::endl;
	//read the points' positions
	for (int i = 0; i < points_numb; i++)
	{
		Vec3d point_in;
		status_in >> point_in[0] >> point_in[1] >> point_in[2];
		centerline_points.push_back(point_in);
	}
	//read the lines
	vector<int> centerline_lines;
	for (int i = 0; i < 2*lines_numb; i++)
	{
		int line_in1,line_in2;
		status_in >> line_in1>>line_in2;
		centerline_lines.push_back(line_in1);
		centerline_lines.push_back(line_in2);
	}
	std::cout << "read centerline successfully" << std::endl;

	//compute points frequency
	Array1i points_frequency;
	points_frequency.resize(points_numb);
	points_frequency.set_zero();
	for (int i = 0; i < 2*lines_numb; i++)
	{
		int line_p = centerline_lines[i];
		points_frequency[line_p] += 1;
	}
	//find the intersections and leaf node
	vector<int> intersection, leaf_node;
	for (int i = 0; i < points_numb; i++)
	{
		if (points_frequency[i] == 1)
			leaf_node.push_back(i);
		else if (points_frequency[i] == 3)
			intersection.push_back(i);
	}

	if (intersection.size() != 2 || leaf_node.size() != 4)
	{
		cout << "the number of intersection or leaf_node is not correct: " << intersection.size() << " " << leaf_node.size() << endl;
		return false;
	}

	//distinguish two intersections
	Vec3d in_direction = (0.457737, -0.447532, -0.768240);
	Vec3d vector_intersection = centerline_points[intersection[0]] - centerline_points[intersection[1]];
	if (dot(in_direction, vector_intersection) > 0)
	{
		outlet_intersection = intersection[1];
		inlet_intersection = intersection[0];
		std::cout << inlet_intersection << " " << outlet_intersection << std::endl;
	}
	else
	{
		outlet_intersection = intersection[0];
		inlet_intersection = intersection[1];
		std::cout << inlet_intersection << " " << outlet_intersection << std::endl;
	}
	//divide the centerline into different branches and distinguish pv,lpv,rpv,sm,svm
	vector<int> line_points1, line_points2, line_points3, line_points4, line_points5;
	int find_point1 = inlet_intersection, find_point2 = inlet_intersection, find_point3 = inlet_intersection, find_point4 = outlet_intersection, find_point5 = outlet_intersection;
	line_points1.push_back(inlet_intersection);
	line_points2.push_back(inlet_intersection);
	line_points3.push_back(inlet_intersection);
	line_points4.push_back(outlet_intersection);
	line_points5.push_back(outlet_intersection);
	auto find_next_point = [&](int i_next, int& find_point_next, vector<int>& line_points_next)
	{//function used to divid branches
		if (find_point_next != outlet_intersection || line_points_next.size() == 1)
		{//if find_point_next=outlet_intersection and line_nodes_next's size>1, we have found pv!
			if (centerline_lines[i_next] == find_point_next)
			{
				if ((i_next % 2) == 0)
				{
					find_point_next = centerline_lines[i_next + 1];
					centerline_lines[i_next] = -1;
					centerline_lines[i_next + 1] = -1;
				}
				else if ((i_next % 2) == 1)
				{
					find_point_next = centerline_lines[i_next - 1];
					centerline_lines[i_next] = -1;
					centerline_lines[i_next - 1] = -1;
				}
				line_points_next.push_back(find_point_next);
			}
		}
	};
	for (int j = 0; j < lines_numb; j++)
	{
		for (int i = 0; i < 2 * lines_numb; i++)
		{//divide branches: pv,smv,sv
			find_next_point(i, find_point1, line_points1);
			find_next_point(i, find_point2, line_points2);
			find_next_point(i, find_point3, line_points3);
		}
	}

	auto distinguish_branches = [&](vector<int>& line_nodes11, vector<int>& line_nodes22)
	{//function used to distinguish smv and sv
		Vec3d smv_direction = (-0.299036, -0.180007, -0.937110);
		Vec3d sv_direction = (0.948883, -0.072825, 0.307111);
		Vec3d direction_in = centerline_points[line_nodes11[1]] - centerline_points[line_nodes11[0]];
		//cout << direction_in << " " << dot(smv_direction, direction_in) << " " << dot(sv_direction, direction_in) << endl;
		if (dot(smv_direction, direction_in) > dot(sv_direction, direction_in))
		{
			smv = line_nodes11;
			sv = line_nodes22;
		}
		else
		{
			smv = line_nodes22;
			sv = line_nodes11;
		}
	};
	if (find_point1 == outlet_intersection)
	{//distinguish pv,smv,sv
		distinguish_branches(line_points2, line_points3);
		pv = line_points1;
	}
	else if (find_point2 == outlet_intersection)
	{
		distinguish_branches(line_points1, line_points3);
		pv = line_points2;
	}
	else if (find_point3 == outlet_intersection)
	{
		distinguish_branches(line_points1, line_points2);
		pv = line_points3;
	}

	for (int j = 0; j < lines_numb; j++)
	{
		for (int i = 0; i < 2 * lines_numb; i++)
		{//divide branches: lpv,rpv
			find_next_point(i, find_point4, line_points4);
			find_next_point(i, find_point5, line_points5);
		}
	}
	//这些用来判断的参数还有问题需要再斟酌一下
	Vec3d lpv_direction(0.051848, -0.819776, 0.0);
	Vec3d rpv_direction(-0.706018, 0.400582, 0.0);
	Vec3d direction_out = centerline_points[line_points4[1]] - centerline_points[line_points4[0]];
	//cout << direction_out<<" "<<lpv_direction<<" "<<dot(direction_out,lpv_direction) << endl;
	//cout << dot(lpv_direction, direction_out) << " " << dot(rpv_direction, direction_out) << endl;
	if (dot(lpv_direction, direction_out) > dot(rpv_direction, direction_out))
	{//distinguish lpv,rpv
		lpv = line_points4;
		rpv = line_points5;
		//cout << direction_out << endl;
	}
	else
	{
		//cout << dot(lpv_direction, direction_out) << endl;
		lpv = line_points5;
		rpv = line_points4;
	}

	std::cout << "the number of pv is: " << pv.size() << std::endl;
	std::cout << "the number of lpv is: " << lpv.size() << std::endl;
	std::cout << "the number of rpv is: " << rpv.size() << std::endl;
	std::cout << "the number of sv is: " << sv.size() << std::endl;
	std::cout << "the number of smv is: " << smv.size() << std::endl;
}

void Simple::initialize_sdf(mfd::DistanceField3D& fSDF, mfd::DistanceField3D& inSDF1, mfd::DistanceField3D& inSDF2, mfd::DistanceField3D& outSDF1, mfd::DistanceField3D& outSDF2, Array3d& p_liq, Array3d& u_liq, Array3d& v_liq, Array3d& w_liq,Array3BT& p_id, Array3BT& u_id, Array3BT& v_id, Array3BT& w_id)
{
	p_liq.resize(ni, nj, nk); u_liq.resize(ni + 1, nj, nk); v_liq.resize(ni, nj + 1, nk); w_liq.resize(ni, nj, nk + 1);
	p_id.resize(ni, nj, nk); u_id.resize(ni + 1, nj, nk); v_id.resize(ni, nj + 1, nk); w_id.resize(ni, nj, nk + 1);

	//Initialize p
	for (int k = 0; k < nk; ++k) for (int j = 0; j < nj; ++j) for (int i = 0; i < ni; ++i)
	{
		Vec3d pos(lower_vertex[0] + (i + 0.5) * dx_mm, lower_vertex[1] + (j + 0.5) * dx_mm, lower_vertex[2] + (k + 0.5) * dx_mm);
		double d;
		fSDF.GetDistance(pos, d);

		if (d < 0)
			p_id(i, j, k) = CellType::Inside;
		else
			p_id(i, j, k) = CellType::Undefined;
		p_liq(i, j, k) = d;
	}

	for (int k = 0; k < nk; ++k) for (int j = 0; j < nj; ++j) for (int i = 0; i < ni; ++i)
	{
		if (p_id(i, j, k) == CellType::Inside)
		{
			Vec3d pos(lower_vertex[0] + (i + 0.5)* dx_mm, lower_vertex[1] + (j + 0.5) * dx_mm, lower_vertex[2] + (k + 0.5) * dx_mm);
			if (p_id(i + 1, j, k) != CellType::Inside)
				classifyBoundary(p_id(i + 1, j, k), pos + Vec3d(dx_mm, 0, 0), inSDF1, inSDF2, outSDF1, outSDF2);

			if (p_id(i - 1, j, k) != CellType::Inside)
				classifyBoundary(p_id(i - 1, j, k), pos + Vec3d(-dx_mm, 0, 0), inSDF1, inSDF2, outSDF1, outSDF2);

			if (p_id(i, j + 1, k) != CellType::Inside)
				classifyBoundary(p_id(i, j + 1, k), pos + Vec3d(0, dx_mm, 0), inSDF1, inSDF2, outSDF1, outSDF2);

			if (p_id(i, j - 1, k) != CellType::Inside)
				classifyBoundary(p_id(i, j - 1, k), pos + Vec3d(0, -dx_mm, 0), inSDF1, inSDF2, outSDF1, outSDF2);

			if (p_id(i, j, k + 1) != CellType::Inside)
				classifyBoundary(p_id(i, j, k + 1), pos + Vec3d(0, 0, dx_mm), inSDF1, inSDF2, outSDF1, outSDF2);

			if (p_id(i, j, k - 1) != CellType::Inside)
				classifyBoundary(p_id(i, j, k - 1), pos + Vec3d(0, 0, -dx_mm), inSDF1, inSDF2, outSDF1, outSDF2);
		}
	}

	//Initialize u
	for (int k = 0; k < nk; ++k) for (int j = 0; j < nj; ++j) for (int i = 0; i < ni + 1; ++i)
	{
		Vec3d pos(lower_vertex[0] + (i)* dx_mm, lower_vertex[1] + (j + 0.5) * dx_mm, lower_vertex[2] + (k + 0.5) * dx_mm);
		double d;
		fSDF.GetDistance(pos, d);
		if (d < 0)
			u_id(i, j, k) = CellType::Inside;
		else
			u_id(i, j, k) = CellType::Undefined;

		u_liq(i, j, k) = d;
	}

	for (int k = 0; k < nk; ++k) for (int j = 0; j < nj; ++j) for (int i = 0; i < ni + 1; ++i)
	{
		if (u_id(i, j, k) == CellType::Inside)
		{
			Vec3d pos(lower_vertex[0] + (i)* dx_mm, lower_vertex[1] + (j + 0.5) * dx_mm, lower_vertex[2] + (k + 0.5) * dx_mm);
			if (u_id(i + 1, j, k) != CellType::Inside)
				classifyBoundary(u_id(i + 1, j, k), pos + Vec3d(dx_mm, 0, 0), inSDF1, inSDF2, outSDF1, outSDF2);

			if (u_id(i - 1, j, k) != CellType::Inside)
				classifyBoundary(u_id(i - 1, j, k), pos + Vec3d(-dx_mm, 0, 0), inSDF1, inSDF2, outSDF1, outSDF2);

			if (u_id(i, j + 1, k) != CellType::Inside)
				classifyBoundary(u_id(i, j + 1, k), pos + Vec3d(0, dx_mm, 0), inSDF1, inSDF2, outSDF1, outSDF2);

			if (u_id(i, j - 1, k) != CellType::Inside)
				classifyBoundary(u_id(i, j - 1, k), pos + Vec3d(0, -dx_mm, 0), inSDF1, inSDF2, outSDF1, outSDF2);

			if (u_id(i, j, k + 1) != CellType::Inside)
				classifyBoundary(u_id(i, j, k + 1), pos + Vec3d(0, 0, dx_mm), inSDF1, inSDF2, outSDF1, outSDF2);

			if (u_id(i, j, k - 1) != CellType::Inside)
				classifyBoundary(u_id(i, j, k - 1), pos + Vec3d(0, 0, -dx_mm), inSDF1, inSDF2, outSDF1, outSDF2);
		}
	}

	//Initialize v
	for (int k = 0; k < nk; ++k) for (int j = 0; j < nj + 1; ++j) for (int i = 0; i < ni; ++i)
	{
		Vec3d pos(lower_vertex[0] + (i + 0.5)* dx_mm, lower_vertex[1] + (j)* dx_mm, lower_vertex[2] + (k + 0.5) * dx_mm);
		double d;
		fSDF.GetDistance(pos, d);
		if (d < 0)
			v_id(i, j, k) = CellType::Inside;
		else
			v_id(i, j, k) = CellType::Undefined;

		v_liq(i, j, k) = d;
	}

	for (int k = 0; k < nk; ++k) for (int j = 0; j < nj + 1; ++j) for (int i = 0; i < ni; ++i)
	{
		if (v_id(i, j, k) == CellType::Inside)
		{
			Vec3d pos(lower_vertex[0] + (i + 0.5) * dx_mm, lower_vertex[1] + (j)* dx_mm, lower_vertex[2] + (k + 0.5) * dx_mm);
			if (v_id(i + 1, j, k) != CellType::Inside)
				classifyBoundary(v_id(i + 1, j, k), pos + Vec3d(dx_mm, 0, 0), inSDF1, inSDF2, outSDF1, outSDF2);

			if (v_id(i - 1, j, k) != CellType::Inside)
				classifyBoundary(v_id(i - 1, j, k), pos + Vec3d(-dx_mm, 0, 0), inSDF1, inSDF2, outSDF1, outSDF2);

			if (v_id(i, j + 1, k) != CellType::Inside)
				classifyBoundary(v_id(i, j + 1, k), pos + Vec3d(0, dx_mm, 0), inSDF1, inSDF2, outSDF1, outSDF2);

			if (v_id(i, j - 1, k) != CellType::Inside)
				classifyBoundary(v_id(i, j - 1, k), pos + Vec3d(0, -dx_mm, 0), inSDF1, inSDF2, outSDF1, outSDF2);

			if (v_id(i, j, k + 1) != CellType::Inside)
				classifyBoundary(v_id(i, j, k + 1), pos + Vec3d(0, 0, dx_mm), inSDF1, inSDF2, outSDF1, outSDF2);

			if (v_id(i, j, k - 1) != CellType::Inside)
				classifyBoundary(v_id(i, j, k - 1), pos + Vec3d(0, 0, -dx_mm), inSDF1, inSDF2, outSDF1, outSDF2);
		}
	}

	//Initialize w
	for (int k = 0; k < nk + 1; ++k) for (int j = 0; j < nj; ++j) for (int i = 0; i < ni; ++i)
	{
		Vec3d pos(lower_vertex[0] + (i + 0.5)* dx_mm, lower_vertex[1] + (j + 0.5)* dx_mm, lower_vertex[2] + (k)* dx_mm);
		double d;
		fSDF.GetDistance(pos, d);
		if (d < 0)
			w_id(i, j, k) = CellType::Inside;
		else
			w_id(i, j, k) = CellType::Undefined;

		w_liq(i, j, k) = d;
	}

	for (int k = 0; k < nk + 1; ++k) for (int j = 0; j < nj; ++j) for (int i = 0; i < ni; ++i)
	{
		if (w_id(i, j, k) == CellType::Inside)
		{
			//double d_inlet, d_outlet;
			Vec3d pos(lower_vertex[0] + (i + 0.5) * dx_mm, lower_vertex[1] + (j + 0.5)* dx_mm, lower_vertex[2] + (k)* dx_mm);
			if (w_id(i + 1, j, k) != CellType::Inside)
				classifyBoundary(w_id(i + 1, j, k), pos + Vec3d(dx_mm, 0, 0), inSDF1, inSDF2, outSDF1, outSDF2);

			if (w_id(i - 1, j, k) != CellType::Inside)
				classifyBoundary(w_id(i - 1, j, k), pos + Vec3d(-dx_mm, 0, 0), inSDF1, inSDF2, outSDF1, outSDF2);

			if (w_id(i, j + 1, k) != CellType::Inside)
				classifyBoundary(w_id(i, j + 1, k), pos + Vec3d(0, dx_mm, 0), inSDF1, inSDF2, outSDF1, outSDF2);

			if (w_id(i, j - 1, k) != CellType::Inside)
				classifyBoundary(w_id(i, j - 1, k), pos + Vec3d(0, -dx_mm, 0), inSDF1, inSDF2, outSDF1, outSDF2);

			if (w_id(i, j, k + 1) != CellType::Inside)
				classifyBoundary(w_id(i, j, k + 1), pos + Vec3d(0, 0, dx_mm), inSDF1, inSDF2, outSDF1, outSDF2);

			if (w_id(i, j, k - 1) != CellType::Inside)
				classifyBoundary(w_id(i, j, k - 1), pos + Vec3d(0, 0, -dx_mm), inSDF1, inSDF2, outSDF1, outSDF2);
		}
	}

	std::string bStr[7] = { {"Undefined"}, {"Inside"}, {"Inlet1"},{"Inlet2"}, {"Outlet1"},{"Outlet2"}, {"Static"} };
	int total_u = 0,total_uin=0;
	for (int k = 0; k < nk; ++k) for (int j = 0; j < nj; ++j) for (int i = 0; i < ni + 1; ++i)
	{
		if (u_id(i, j, k) != CellType::Undefined)
		{
			total_u++;
			if (u_id(i, j, k) == CellType::Inside) total_uin++;
			//std::printf("u: %d %d %d: %s %f \n", i, j, k, bStr[u_id(i, j, k)].c_str(), u_liq(i, j, k));
		}
	}

	int total_v = 0,total_vin=0;
	for (int k = 0; k < nk; ++k) for (int j = 0; j < nj + 1; ++j) for (int i = 0; i < ni; ++i)
	{
		if (v_id(i, j, k) != CellType::Undefined)
		{
			total_v++;
			if (v_id(i, j, k) == CellType::Inside) total_vin++;
			//std::printf("v: %d %d %d: %s %f \n", i, j, k, bStr[v_id(i, j, k)].c_str(),v_liq(i,j,k));
		}
	}

	int total_w = 0,total_win=0;
	for (int k = 0; k < nk + 1; ++k) for (int j = 0; j < nj; ++j) for (int i = 0; i < ni; ++i)
	{
		if (w_id(i, j, k)!= CellType::Undefined)
		{
			total_w++;
			if (w_id(i, j, k) == CellType::Inside) total_win++;
			//std::printf("w: %d %d %d: %s %f \n", i, j, k, bStr[w_id(i, j, k)].c_str(),w_liq(i,j,k)); 
		}
	}

	int p_num = 0,pin_num=0;
	for (int k = 0; k < nk; ++k) for (int j = 0; j < nj; ++j) for (int i = 0; i < ni; ++i)
	{
		if (p_id(i, j, k) != CellType::Undefined)
		{
			p_num++;
			if (p_id(i, j, k) == CellType::Inside) pin_num++;
			//std::printf("p: %d %d %d: %s %f \n", i, j, k, bStr[p_id(i, j, k)].c_str(),p_liq(i,j,k));
		}
	}
	std::cout << "total u number: " << total_uin << "  " << total_u << endl;
	std::cout << "total v number: " << total_vin << "  " << total_v << endl;
	std::cout << "total w number: " << total_win << "  " << total_w << endl;
	std::cout << "total p number: " << pin_num << "  " << p_num << endl;
	
	std::cout << "initialize successful" << endl;
}

void Simple::initialize_sdf(mfd::DistanceField3D& fSDF)
{
	//find the inlet point and compute inlet direction
	double inlet_dist = 0;
	Vec3d inlet_point;

	for (int i = 0; i < pv.size()-1; i++)
	{
		double inlet_dist_next = dist(centerline_points[pv[i]], centerline_points[pv[i + 1]]);
		inlet_dist += inlet_dist_next;
		std::cout << "the long of inlet is: " << inlet_dist << std::endl;
		if (inlet_dist >= inlet_standard)
		{
			inlet_direction1 = inlet_direction2 = normalized(centerline_points[pv[i]] - centerline_points[pv[i + 1]]);
			double inlet_diff = inlet_dist - inlet_standard;
			inlet_point = lerp(centerline_points[pv[i + 1]], centerline_points[pv[i]], inlet_diff / inlet_dist_next);
			std::cout << "inlet direction and inlet point is: " << inlet_direction1 << " " << inlet_point << std::endl;
			break;
		}
	}
	std::cout << "it is ok here now1" << std::endl;

	//find the outlet point and compute outlet direction of LPV and RPV
	double outlet_dist1 = 0, outlet_dist2 = 0;
	Vec3d outlet_point1, outlet_point2;
	for (int i = 0; i < lpv.size()-1; i++)
	{
		double outlet_dist_next = dist(centerline_points[lpv[i]], centerline_points[lpv[i + 1]]);
		outlet_dist1 += outlet_dist_next;
		std::cout << "the long of outlet1 is: " << outlet_dist1 << std::endl;
		if (outlet_dist1 >= outlet_standard1)
		{
			outlet_direction1 = normalized(centerline_points[lpv[i + 1]] - centerline_points[lpv[i]]);
			double outlet_diff = outlet_dist1 - outlet_standard1;
			outlet_point1 = lerp(centerline_points[lpv[i + 1]], centerline_points[lpv[i]], outlet_diff / outlet_dist_next);
			std::cout << "outlet direction1 and outlet point1 is: " << outlet_direction1 << " " << outlet_point1 << std::endl;
			break;
		}
	}
	std::cout << "it is ok here now2" << std::endl;

	for (int i = 0; i < rpv.size()-1; i++)
	{
		double outlet_dist_next = dist(centerline_points[rpv[i]], centerline_points[rpv[i + 1]]);
		outlet_dist2 += outlet_dist_next;
		std::cout << "the long of outlet2 is: " << outlet_dist2 << std::endl;
		if (outlet_dist2 >= outlet_standard2)
		{
			outlet_direction2 = normalized(centerline_points[rpv[i + 1]] - centerline_points[rpv[i]]);
			double outlet_diff = outlet_dist2 - outlet_standard2;
			outlet_point2 = lerp(centerline_points[rpv[i + 1]], centerline_points[rpv[i]], outlet_diff / outlet_dist_next);
			std::cout << "outlet direction2 and outlet point2 is: " << outlet_direction2 << " " << outlet_point2 << std::endl;
			break;
		}
	}
	std::cout << "it is ok here now3" << std::endl;

	//compute sdf
	liquid_phi.resize(ni, nj, nk); u_liquid_phi.resize(ni + 1, nj, nk); v_liquid_phi.resize(ni, nj + 1, nk); w_liquid_phi.resize(ni, nj, nk + 1);
	p_identifer.resize(ni, nj, nk); u_identifer.resize(ni + 1, nj, nk); v_identifer.resize(ni, nj + 1, nk); w_identifer.resize(ni, nj, nk + 1);
	std::cout << "it is ok here now3" << std::endl;

	auto classify_boundary = [&](Vec3d pos, CellType& identifer, double& liq_phi)
	{//function used to classify boundary
		double d;
		fSDF.GetDistance(pos, d);
		if (d < 0)
		{
			double d1 = dot(pos, inlet_direction1) - dot(inlet_point, inlet_direction1);
			double d2 = dot(pos, outlet_direction1) - dot(outlet_point1, outlet_direction1);
			double d3 = dot(pos, outlet_direction2) - dot(outlet_point2, outlet_direction2);

			if ((d1 < 0 && d2 < 0) && d3 < 0)
			{
				identifer = CellType::Inside;
				liq_phi = max(max(max(d, d1), d2), d3);
			}
			else if (d1 > 0)
			{
				identifer = CellType::Inlet1;
				liq_phi = d1;
			}
			else if (d2 > 0)
			{
				identifer = CellType::Outlet1;
				liq_phi = d2;
			}
			else if (d3 > 0)
			{
				identifer = CellType::Outlet2;
				liq_phi = d3;
			}
		}
		else
		{
			identifer = CellType::Static;
			liq_phi = d;
		}
	};

	//Initialize p
	for (int k = 0; k < nk; ++k) 
		for (int j = 0; j < nj; ++j)
			for (int i = 0; i < ni; ++i)
			{
				Vec3d pos(lower_vertex[0] + (i + 0.5) * dx_mm, lower_vertex[1] + (j + 0.5) * dx_mm, lower_vertex[2] + (k + 0.5) * dx_mm);
				classify_boundary(pos, p_identifer(i, j, k), liquid_phi(i, j, k));
			}
	std::cout << "it is ok here now4" << std::endl;

	//Initialize u
	for (int k = 0; k < nk; ++k)
		for (int j = 0; j < nj; ++j)
			for (int i = 0; i < ni+1; ++i)
			{
				Vec3d pos(lower_vertex[0] + i*dx_mm, lower_vertex[1] + (j + 0.5) * dx_mm, lower_vertex[2] + (k + 0.5) * dx_mm);
				classify_boundary(pos, u_identifer(i, j, k), u_liquid_phi(i, j, k));
			}
	std::cout << "it is ok here now5" << std::endl;

	//Initialize v
	for (int k = 0; k < nk; ++k)
		for (int j = 0; j < nj+1; ++j)
			for (int i = 0; i < ni; ++i)
			{
				Vec3d pos(lower_vertex[0] + (i + 0.5) * dx_mm, lower_vertex[1] + j * dx_mm, lower_vertex[2] + (k + 0.5) * dx_mm);
				classify_boundary(pos, v_identifer(i, j, k), v_liquid_phi(i, j, k));
			}
	std::cout << "it is ok here now6" << std::endl;

	//Initialize w
	for (int k = 0; k < nk+1; ++k)
		for (int j = 0; j < nj; ++j)
			for (int i = 0; i < ni; ++i)
			{
				Vec3d pos(lower_vertex[0] + (i + 0.5) * dx_mm, lower_vertex[1] + (j + 0.5) * dx_mm, lower_vertex[2] + k * dx_mm);
				classify_boundary(pos, w_identifer(i, j, k), w_liquid_phi(i, j, k));
			}
}

void Simple::advance(std::string path, int frame, double dt, double alpha,double theta)
{
	std::cout << "**************************Frame: " << frame << " Started**************************" << std::endl;
	
	double t = 0.0;
	int nSubstep = 1;
	while (t < dt)
	{
		std::cout << std::endl;
		std::cout << "----------------Substep: " << nSubstep << "----------------" << std::endl;

		double substep1 = cfl();
		double substep = 0.01;

		if (t + substep > dt)
			substep = dt - t;
		std::printf("Taking substep of size %f (to %0.3f%% of the frame)\n", substep, 100 * (t + substep) / dt);
		//total_time += substep;
		//compute_velocity(total_time);

		//printf(" Extrapolation\n");
		//Array3BT u_id = u_identifer;
		//Array3BT v_id = v_identifer;
		//Array3BT w_id = w_identifer;
		//extrapolate(u, u_id);
		//extrapolate(v, v_id);
		//extrapolate(w, w_id);
		////Advection
		//printf(" Velocity advection\n");
		//advect(u, v, w, substep);

		//int n_iterations = 1;
		//convergence_indicator = 10;
		//while ((convergence_indicator > convergence_standard) && (n_iterations < number_iterations))
		//{
		//	cout << "the convergence indicator of simple iteration is: " << convergence_indicator << endl;
		//	std::cout << "^^^^^^^Starting " << n_iterations << "th iteration of simple^^^^^^^" << std::endl;

		//	u_starstar.copyFrom(u);
		//	v_starstar.copyFrom(v);
		//	w_starstar.copyFrom(w);
		//	update_velocity(u_starstar, v_starstar, w_starstar, p_star, substep, alpha,theta);
		//	solve_velocity_component(u_prime, u_starstar, u_identifer, u_liquid_phi, substep, alpha,theta);
		//	solve_velocity_component(v_prime, v_starstar, v_identifer, v_liquid_phi, substep, alpha,theta);
		//	solve_velocity_component(w_prime, w_starstar, w_identifer, w_liquid_phi, substep, alpha,theta);

		//	solve_pressure(p_prime, u_prime, v_prime, w_prime, substep, alpha,theta);

		//	for (int i = 0; i < p_prime.size(); i++)
		//	{
		//		p_star[i] += p_prime[i];
		//	}

		//	update_velocity(u_prime, v_prime, w_prime, p_prime, substep, alpha,theta);

		//	std::cout << std::endl;

		//	n_iterations++;
		//}
		//u.copyFrom(u_prime);
		//v.copyFrom(v_prime);
		//w.copyFrom(w_prime);
		//p.copyFrom(p_star);

		//pressure_indicator = abs(statistics_pressure(face0) - pressure0);
		//pressure0 = statistics_pressure(face0);
		//std::cout << "the pressure_indicator is: " << pressure_indicator << endl;

		int n_iterations = 1;
		convergence_indicator = 10;
		while ((convergence_indicator > convergence_standard) && (n_iterations < number_iterations))
		{
			cout << "the convergence indicator of simple iteration is: " << convergence_indicator << endl;
			std::cout << "^^^^^^^Starting " << n_iterations << "th iteration of simple^^^^^^^" << std::endl;

			update_velocity(u_starstar, v_starstar, w_starstar, p_star, substep, alpha, theta);
			solve_velocity_component(u_prime, u_starstar, u_identifer, u_liquid_phi, substep, alpha, theta);
			solve_velocity_component(v_prime, v_starstar, v_identifer, v_liquid_phi, substep, alpha, theta);
			solve_velocity_component(w_prime, w_starstar, w_identifer, w_liquid_phi, substep, alpha, theta);

			solve_pressure(p_prime, u_prime, v_prime, w_prime, substep, alpha, theta);

			for (int i = 0; i < p_prime.size(); i++)
			{
				p_star[i] += p_prime[i];
			}

			u_starstar.copyFrom(u_prime);
			v_starstar.copyFrom(v_prime);
			w_starstar.copyFrom(w_prime);
			update_velocity(u_starstar, v_starstar, w_starstar, p_prime, substep, alpha, theta);

			u.copyFrom(u_starstar);
			v.copyFrom(v_starstar);
			w.copyFrom(w_starstar);
			double substep2 = cfl();

			p.copyFrom(p_star);
			//pressure_indicator = abs(statistics_pressure(face0) - pressure0);
			//pressure0 = statistics_pressure(face0);
			//std::cout << "the pressure_indicator is: " << pressure_indicator << endl;

			std::cout << std::endl;
			n_iterations++;
		}
		//for (int k = 0; k < nk; ++k) for (int j = 0; j < nj; ++j) for (int i = 0; i < ni; ++i)
		//{
		//	if (p_identifer(i, j, k) == CellType::Inside)
		//		std::printf("p: %d %d %d: %f %f \n", i, j, k, liquid_phi(i,j,k),p(i, j, k));
		//}
		t += substep;
		std::cout << std::endl << std::endl;
		nSubstep++;
	}
}

#ifdef BLOCK
void Simple::solve_velocity_component(Array3d& vel, Array3d& vel_ss, Array3BT& identifier, Array3d& vel_phi, double dt, double alpha, double theta)
{
	int system_size = vel.ni * vel.nj * vel.nk;
	b.resize(system_size);
	A.resize(system_size, system_size);
	std::vector<TripletD> coefficients;
	Eigen::VectorXd x(system_size);

	b.setZero();

	double term = kinetic_coefficient * dt / (blood_density * dx * dx);
	auto setupNeighbors = [&](
		double& s_i,
		int i,
		int i_minus,
		int i_plus,
		double phi_i_minus,
		double phi_i_plus,
		CellType cell_i_minus,
		CellType cell_i_plus,
		double vb_1_plus,
		double vb_1_minus,
		double vb_2_plus,
		double vb_2_minus)
	{
		if (cell_i_minus == CellType::Inside && cell_i_plus == CellType::Inside)
		{
			coefficients.push_back(TripletD(i, i_minus, -term));
			coefficients.push_back(TripletD(i, i_plus, -term));
			coefficients.push_back(TripletD(i, i, 2 * term));
		}
		else if (cell_i_minus == CellType::Inside && cell_i_plus != CellType::Inside)
		{
			if (cell_i_plus == CellType::Static|| cell_i_plus == CellType::Outlet2)
			{
				double A = phi_i_plus / (2 * dx_mm - phi_i_plus);
				coefficients.push_back(TripletD(i, i_minus, -term * (1 - A)));
				coefficients.push_back(TripletD(i, i, 2 * term));
			}
			else if (cell_i_plus == CellType::Inlet1)
			{
				double A = phi_i_plus / (2 * dx_mm - phi_i_plus);
				coefficients.push_back(TripletD(i, i_minus, -term * (1 - A)));
				coefficients.push_back(TripletD(i, i, 2 * term));

				s_i += term * (1 + A) * vb_1_plus;
			}
			else if (cell_i_plus == CellType::Inlet2)
			{
				double A = phi_i_plus / (2 * dx_mm - phi_i_plus);
				coefficients.push_back(TripletD(i, i_minus, -term * (1 - A)));
				coefficients.push_back(TripletD(i, i, 2 * term));

				s_i += term * (1 + A) * vb_2_plus;
			}
			else if(cell_i_plus==CellType::Outlet1)
			{
				coefficients.push_back(TripletD(i, i_minus, -term));
				coefficients.push_back(TripletD(i, i, term));
			}
		}
		else if (cell_i_minus != CellType::Inside && cell_i_plus == CellType::Inside)
		{
			if (cell_i_minus == CellType::Static|| cell_i_minus == CellType::Outlet2)
			{
				double A = phi_i_minus / (2 * dx_mm - phi_i_minus);
				coefficients.push_back(TripletD(i, i_plus, -term * (1 - A)));
				coefficients.push_back(TripletD(i, i, 2 * term));
			}
			else if (cell_i_minus == CellType::Inlet1)
			{
				double A = phi_i_minus / (2 * dx_mm - phi_i_minus);
				coefficients.push_back(TripletD(i, i_plus, -term * (1 - A)));
				coefficients.push_back(TripletD(i, i, 2 * term));

				s_i += term * (1 + A) * vb_1_minus;
			}
			else if (cell_i_minus == CellType::Inlet2)
			{
				double A = phi_i_minus / (2 * dx_mm - phi_i_minus);
				coefficients.push_back(TripletD(i, i_plus, -term * (1 - A)));
				coefficients.push_back(TripletD(i, i, 2 * term));

				s_i += term * (1 + A) * vb_2_minus;
			}
			else if (cell_i_minus == CellType::Outlet1)
			{
				coefficients.push_back(TripletD(i, i_plus, -term));
				coefficients.push_back(TripletD(i, i, term));
			}
		}
		else
		{
			//TODO: Evaluate how to handle this case

			//double theta = fraction_inside(phi_i, phi_j);
			//if (theta < 0.01f) theta = 0.01f;
			////coefficients.push_back(TripletD(i, j_opposite, d*(1 - theta) / (theta + 1)));
			//coefficients.push_back(TripletD(p_i, p_i, term / theta));
		}
	};

	for (int k = 0; k < vel.nk; ++k) {
		for (int j = 0; j < vel.nj; ++j) {
			for (int i = 0; i < vel.ni; ++i) {
				int index = vel.index(i, j, k);
				if (identifier(i, j, k) == CellType::Inside)
				{
					double& b_ijk = b(index);
					b(index) += vel_ss(i, j, k);
					coefficients.push_back(TripletD(index, index, 1.0));
					if (vel.ni > ni)
					{
						setupNeighbors(b_ijk, index, vel.index(i - 1, j, k), vel.index(i + 1, j, k), vel_phi(i - 1, j, k), vel_phi(i + 1, j, k), identifier(i - 1, j, k), identifier(i + 1, j, k), u_boundary(inlet_direction1, inlet_magnitude1, alpha, theta, j, k), u_boundary(inlet_direction1, inlet_magnitude1, alpha, theta, j, k), u_boundary(inlet_direction2, inlet_magnitude2, alpha, theta, j, k), u_boundary(inlet_direction2, inlet_magnitude2, alpha, theta, j, k));

						setupNeighbors(b_ijk, index, vel.index(i, j - 1, k), vel.index(i, j + 1, k), vel_phi(i, j - 1, k), vel_phi(i, j + 1, k), identifier(i, j - 1, k), identifier(i, j + 1, k), u_boundary(inlet_direction1, inlet_magnitude1, alpha, theta, j + 1, k), u_boundary(inlet_direction1, inlet_magnitude1, alpha, theta, j - 1, k), u_boundary(inlet_direction2, inlet_magnitude2, alpha, theta, j + 1, k), u_boundary(inlet_direction2, inlet_magnitude2, alpha, theta, j - 1, k));

						setupNeighbors(b_ijk, index, vel.index(i, j, k - 1), vel.index(i, j, k + 1), vel_phi(i, j, k - 1), vel_phi(i, j, k + 1), identifier(i, j, k - 1), identifier(i, j, k + 1), u_boundary(inlet_direction1, inlet_magnitude1, alpha, theta, j, k + 1), u_boundary(inlet_direction1, inlet_magnitude1, alpha, theta, j, k - 1), u_boundary(inlet_direction2, inlet_magnitude2, alpha, theta, j, k + 1), u_boundary(inlet_direction2, inlet_magnitude2, alpha, theta, j, k - 1));
					}
					else if (vel.nj > nj)
					{
						setupNeighbors(b_ijk, index, vel.index(i - 1, j, k), vel.index(i + 1, j, k), vel_phi(i - 1, j, k), vel_phi(i + 1, j, k), identifier(i - 1, j, k), identifier(i + 1, j, k), v_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), v_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), v_boundary(inlet_direction2, inlet_magnitude2, alpha, theta), v_boundary(inlet_direction2, inlet_magnitude2, alpha, theta));

						setupNeighbors(b_ijk, index, vel.index(i, j - 1, k), vel.index(i, j + 1, k), vel_phi(i, j - 1, k), vel_phi(i, j + 1, k), identifier(i, j - 1, k), identifier(i, j + 1, k), v_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), v_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), v_boundary(inlet_direction2, inlet_magnitude2, alpha, theta), v_boundary(inlet_direction2, inlet_magnitude2, alpha, theta));

						setupNeighbors(b_ijk, index, vel.index(i, j, k - 1), vel.index(i, j, k + 1), vel_phi(i, j, k - 1), vel_phi(i, j, k + 1), identifier(i, j, k - 1), identifier(i, j, k + 1), v_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), v_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), v_boundary(inlet_direction2, inlet_magnitude2, alpha, theta), v_boundary(inlet_direction2, inlet_magnitude2, alpha, theta));
					}
					else if (vel.nk > nk)
					{
						setupNeighbors(b_ijk, index, vel.index(i - 1, j, k), vel.index(i + 1, j, k), vel_phi(i - 1, j, k), vel_phi(i + 1, j, k), identifier(i - 1, j, k), identifier(i + 1, j, k), w_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), w_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), w_boundary(inlet_direction2, inlet_magnitude2, alpha, theta), w_boundary(inlet_direction2, inlet_magnitude2, alpha, theta));

						setupNeighbors(b_ijk, index, vel.index(i, j - 1, k), vel.index(i, j + 1, k), vel_phi(i, j - 1, k), vel_phi(i, j + 1, k), identifier(i, j - 1, k), identifier(i, j + 1, k), w_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), w_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), w_boundary(inlet_direction2, inlet_magnitude2, alpha, theta), w_boundary(inlet_direction2, inlet_magnitude2, alpha, theta));

						setupNeighbors(b_ijk, index, vel.index(i, j, k - 1), vel.index(i, j, k + 1), vel_phi(i, j, k - 1), vel_phi(i, j, k + 1), identifier(i, j, k - 1), identifier(i, j, k + 1), w_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), w_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), w_boundary(inlet_direction2, inlet_magnitude2, alpha, theta), w_boundary(inlet_direction2, inlet_magnitude2, alpha, theta));
					}
				}
			}
		}
	}

	A.setFromTriplets(coefficients.begin(), coefficients.end());
	Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> so;
	so.compute(A);
	x = so.solve(b);
	std::cout << "#iterations:   " << so.iterations() << "    #estimated error:   " << so.error() << std::endl;

	//update velocity
	for (int k = 0; k < vel.nk; ++k) {
		for (int j = 0; j < vel.nj; ++j) {
			for (int i = 0; i < vel.ni; ++i) {
				int index = vel.index(i, j, k);
				if (identifier(i, j, k) == CellType::Inside)
					vel(i, j, k) = x(index);
			}
		}
	}
}

void Simple::solve_pressure(Array3d& pressure, Array3d& u_in, Array3d& v_in, Array3d& w_in, double dt, double alpha, double theta)
{
	int system_size = pressure.ni * pressure.nj * pressure.nk;
	b.resize(system_size);
	A.resize(system_size, system_size);
	std::vector<TripletD> coefficients;
	Eigen::VectorXd x(system_size);

	b.setZero();

	pressure.set_zero();

	double term = dx * blood_density / dt;
	auto setupPPE = [&](
		double& s_i,
		int p_i,
		int p_i_minus,
		int p_i_plus,
		double v_i_minus,
		double v_i_plus,
		double v_phi_i_minus,
		double v_phi_i_plus,
		CellType v_type_i_minus,
		CellType v_type_i_plus,
		double vb_1_plus,
		double vb_1_minus,
		double vb_2_plus,
		double vb_2_minus,
		double pb_1)
	{
		CellType p_type_i_minus = p_identifer[p_i_minus];
		CellType p_type_i_plus = p_identifer[p_i_plus];

		double p_phi_i_plus = liquid_phi[p_i_plus];
		double p_phi_i_minus = liquid_phi[p_i_minus];

		if (p_type_i_minus == CellType::Inside && p_type_i_plus == CellType::Inside)
		{
			coefficients.push_back(TripletD(p_i, p_i_plus, -1.0));
			coefficients.push_back(TripletD(p_i, p_i_minus, -1.0));
			coefficients.push_back(TripletD(p_i, p_i, 2.0));

			s_i += -term * (v_i_plus - v_i_minus);
		}
		else if (p_type_i_minus == CellType::Inside && p_type_i_plus != CellType::Inside)
		{
			//TODO: add contribution from boundary velocity
			if (p_type_i_plus == CellType::Static|| p_type_i_plus == CellType::Outlet2)
			{
				//Impose velocity boundary conditions
				double A = -dx_mm / (v_phi_i_plus - dx_mm);
				coefficients.push_back(TripletD(p_i, p_i_minus, -A));
				coefficients.push_back(TripletD(p_i, p_i, A));

				s_i += term * v_i_minus * A;
			}
			else if (p_type_i_plus == CellType::Inlet1)
			{
				//Impose velocity boundary conditions
				double A = -dx_mm / (v_phi_i_plus - dx_mm);
				coefficients.push_back(TripletD(p_i, p_i_minus, -A));
				coefficients.push_back(TripletD(p_i, p_i, A));

				s_i += term * v_i_minus * A;
				//TODO: compute the inlet velocity boundary
				s_i += -term * vb_1_plus * A;
			}
			else if (p_type_i_plus == CellType::Inlet2)
			{
				//Impose velocity boundary conditions
				double A = -dx_mm / (v_phi_i_plus - dx_mm);
				coefficients.push_back(TripletD(p_i, p_i_minus, -A));
				coefficients.push_back(TripletD(p_i, p_i, A));

				s_i += term * v_i_minus * A;
				//TODO: compute the inlet velocity boundary
				s_i += -term * vb_2_plus * A;
			}
			else if (p_type_i_plus == CellType::Outlet1)
			{
				//TODO: check the solvability
				if (v_type_i_plus == CellType::Inside)
				{
					//Impose pressure boundary conditions
					double A = p_phi_i_plus / (2 * dx_mm - p_phi_i_plus);

					coefficients.push_back(TripletD(p_i, p_i_minus, -(1 - A)));
					coefficients.push_back(TripletD(p_i, p_i, 2.0));

					//TODO: check the sign for this term
					s_i += -term * (v_i_plus - v_i_minus);

					//TODO: check the sign for this term
					s_i += (1 + A) * pb_1;
				}
			}
		}
		else if (p_type_i_minus != CellType::Inside && p_type_i_plus == CellType::Inside)
		{
			if (p_type_i_minus == CellType::Static|| p_type_i_minus == CellType::Outlet2)
			{
				//Impose velocity boundary conditions
				double A = -dx_mm / (v_phi_i_minus - dx_mm);

				coefficients.push_back(TripletD(p_i, p_i_plus, -A));
				coefficients.push_back(TripletD(p_i, p_i, A));

				s_i += -term * v_i_plus * A;
			}
			else if (p_type_i_minus == CellType::Inlet1)
			{
				//Impose velocity boundary conditions
				double A = -dx_mm / (v_phi_i_minus - dx_mm);

				coefficients.push_back(TripletD(p_i, p_i_plus, -A));
				coefficients.push_back(TripletD(p_i, p_i, A));

				s_i += -term * v_i_plus * A;

				s_i += term * vb_1_minus * A;
			}
			else if (p_type_i_minus == CellType::Inlet2)
			{
				//Impose velocity boundary conditions
				double A = -dx_mm / (v_phi_i_minus - dx_mm);

				coefficients.push_back(TripletD(p_i, p_i_plus, -A));
				coefficients.push_back(TripletD(p_i, p_i, A));

				s_i += -term * v_i_plus * A;

				s_i += term * vb_2_minus * A;
			}
			else if (p_type_i_minus == CellType::Outlet1)
			{
				//TODO: check the solvability
				if (v_type_i_minus == CellType::Inside)
				{
					//Impose pressure boundary conditions
					double A = p_phi_i_minus / (2 * dx_mm - p_phi_i_minus);
					coefficients.push_back(TripletD(p_i, p_i_plus, -(1 - A)));
					coefficients.push_back(TripletD(p_i, p_i, 2.0));

					//TODO: check the sign for this term
					s_i += -term * (v_i_plus - v_i_minus);

					//TODO: check the sign for this term
					s_i += (1 + A) * pb_1;
				}
			}
		}
		else
		{
			//TODO: Evaluate how to handle this case

// 			double theta = fraction_inside(phi_i, phi_j);
// 			if (theta < 0.01f) theta = 0.01f;
// 			//coefficients.push_back(TripletD(i, j_opposite, d*(1 - theta) / (theta + 1)));
// 			coefficients.push_back(TripletD(p_i, p_i, term / theta));
		}
	};

	for (int k = 0; k < nk; ++k) {
		for (int j = 0; j < nj; ++j) {
			for (int i = 0; i < ni; ++i) {
				int index = pressure.index(i, j, k);
				if (p_identifer(i, j, k) == CellType::Inside)
				{
					double source_ijk = 0.0;
					setupPPE(source_ijk, index, pressure.index(i - 1, j, k), pressure.index(i + 1, j, k), u_in(i, j, k), u_in(i + 1, j, k), u_liquid_phi(i, j, k), u_liquid_phi(i + 1, j, k), u_identifer(i, j, k), u_identifer(i + 1, j, k), u_boundary(inlet_direction1, inlet_magnitude1, alpha, theta, j, k), u_boundary(inlet_direction1, inlet_magnitude1, alpha, theta, j, k), u_boundary(inlet_direction2, inlet_magnitude2, alpha, theta, j, k), u_boundary(inlet_direction2, inlet_magnitude2, alpha, theta, j, k), p_boundary());

					setupPPE(source_ijk, index, pressure.index(i, j - 1, k), pressure.index(i, j + 1, k), v_in(i, j, k), v_in(i, j + 1, k), v_liquid_phi(i, j, k), v_liquid_phi(i, j + 1, k), v_identifer(i, j, k), v_identifer(i, j + 1, k), v_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), v_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), v_boundary(inlet_direction2, inlet_magnitude2, alpha, theta), v_boundary(inlet_direction2, inlet_magnitude2, alpha, theta), p_boundary());

					setupPPE(source_ijk, index, pressure.index(i, j, k - 1), pressure.index(i, j, k + 1), w_in(i, j, k), w_in(i, j, k + 1), w_liquid_phi(i, j, k), w_liquid_phi(i, j, k + 1), w_identifer(i, j, k), w_identifer(i, j, k + 1), w_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), w_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), w_boundary(inlet_direction2, inlet_magnitude2, alpha, theta), w_boundary(inlet_direction2, inlet_magnitude2, alpha, theta), p_boundary());

					b[index] = source_ijk;
				}
			}
		}
	}
	A.setFromTriplets(coefficients.begin(), coefficients.end());
	Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> so;
	so.compute(A);
	x = so.solve(b);
	std::cout << "Pressure iterations:   " << so.iterations() << "    #estimated error:   " << so.error() << std::endl;

	//update velocity
	for (int k = 0; k < nk; ++k) {
		for (int j = 0; j < nj; ++j) {
			for (int i = 0; i < ni; ++i) {
				int index = pressure.index(i, j, k);
				if (p_identifer(i, j, k) == CellType::Inside)
					pressure(i, j, k) = x(index);
			}
		}
	}
	convergence_indicator = b.norm();
}

void Simple::update_velocity(Array3d& u, Array3d& v, Array3d& w, Array3d& pressure, double dt, double alpha, double theta)
{
	std::cout << "************this conditation is BLOCK" << std::endl;
	double term = dt / (dx * blood_density);

	for (int k = 0; k < u.nk; ++k) for (int j = 0; j < u.nj; ++j) for (int i = 1; i < u.ni - 1; ++i)
	{
		if (u_identifer(i, j, k) == CellType::Inside)
		{
			if (p_identifer(i - 1, j, k) == CellType::Inside && p_identifer(i, j, k) == CellType::Inside)
				u(i, j, k) -= term * (pressure(i, j, k) - pressure(i - 1, j, k));
		}
	}
	for (int k = 0; k < v.nk; ++k) for (int j = 1; j < v.nj - 1; ++j) for (int i = 0; i < v.ni; ++i)
	{
		if (v_identifer(i, j, k) == CellType::Inside)
		{
			if (p_identifer(i, j - 1, k) == CellType::Inside && p_identifer(i, j, k) == CellType::Inside)
				v(i, j, k) -= term * (pressure(i, j, k) - pressure(i, j - 1, k));
		}
	}
	for (int k = 1; k < w.nk - 1; ++k) for (int j = 0; j < w.nj; ++j) for (int i = 0; i < w.ni; ++i)
	{
		if (w_identifer(i, j, k) == CellType::Inside)
		{
			if (p_identifer(i, j, k - 1) == CellType::Inside && p_identifer(i, j, k) == CellType::Inside)
				w(i, j, k) -= term * (pressure(i, j, k) - pressure(i, j, k - 1));
		}
	}

	auto setupBoundaryVelocity = [&](
		double& v_i_minus,
		double& v_i_plus,
		int p_i,
		int p_i_minus,
		int p_i_plus,
		double v_phi_i_minus,
		double v_phi_i_plus,
		CellType v_type_i_minus,
		CellType v_type_i_plus,
		double vb_1_plus,
		double vb_1_minus,
		double vb_2_plus,
		double vb_2_minus,
		double pb_1)
	{
		CellType p_type_i_minus = p_identifer[p_i_minus];
		CellType p_type_i_plus = p_identifer[p_i_plus];

		double p_phi_i_minus = liquid_phi[p_i_minus];
		double p_phi_i_plus = liquid_phi[p_i_plus];

		if (p_type_i_minus == CellType::Inside && p_type_i_plus != CellType::Inside)
		{
			if (v_type_i_plus == CellType::Inside)
			{
				//TODO: add contribution from boundary velocity
				if (p_type_i_plus == CellType::Static|| p_type_i_plus == CellType::Outlet2)
				{
					double A = -dx_mm / (v_phi_i_plus - dx_mm);

					v_i_plus = (1 - A) * v_i_minus;
				}
				else if (p_type_i_plus == CellType::Inlet1)
				{
					double A = -dx_mm / (v_phi_i_plus - dx_mm);

					v_i_plus = (1 - A) * v_i_minus + A * vb_1_plus;
				}
				else if (p_type_i_plus == CellType::Inlet2)
				{
					double A = -dx_mm / (v_phi_i_plus - dx_mm);

					v_i_plus = (1 - A) * v_i_minus + A * vb_2_plus;
				}
				else if (p_type_i_plus == CellType::Outlet1)
				{
					double A = p_phi_i_plus / (2 * dx_mm - p_phi_i_plus);

					double pressure_i_plus_new = -A * pressure[p_i_minus] + (1 + A) * pb_1;
					v_i_plus -= term * (pressure_i_plus_new - pressure[p_i]);
				}
			}
		}
		else if (p_type_i_minus != CellType::Inside && p_type_i_plus == CellType::Inside)
		{
			if (v_type_i_minus == CellType::Inside)
			{
				if (p_type_i_minus == CellType::Static|| p_type_i_minus == CellType::Outlet2)
				{
					double A = -dx_mm / (v_phi_i_minus - dx_mm);

					v_i_minus = (1 - A) * v_i_plus;
				}
				else if (p_type_i_minus == CellType::Inlet1)
				{
					double A = -dx_mm / (v_phi_i_minus - dx_mm);

					v_i_minus = (1 - A) * v_i_plus + A * vb_1_minus;
				}
				else if (p_type_i_minus == CellType::Inlet2)
				{
					double A = -dx_mm / (v_phi_i_minus - dx_mm);

					v_i_minus = (1 - A) * v_i_plus + A * vb_2_minus;
				}
				else if (p_type_i_minus == CellType::Outlet1)
				{
					double A = p_phi_i_minus / (2 * dx_mm - p_phi_i_minus);

					double pressure_i_minus_new = -A * pressure[p_i_plus] + (1 + A) * pb_1;
					v_i_minus -= term * (pressure[p_i] - pressure_i_minus_new);
				}
			}
		}
	};
	//Impose boundary conditions
	for (int k = 0; k < pressure.nk; ++k) for (int j = 0; j < pressure.nj; ++j) for (int i = 0; i < pressure.ni; ++i)
	{
		if (p_identifer(i, j, k) == CellType::Inside)
		{
			setupBoundaryVelocity(u(i, j, k), u(i + 1, j, k), pressure.index(i, j, k), pressure.index(i - 1, j, k), pressure.index(i + 1, j, k), u_liquid_phi(i, j, k), u_liquid_phi(i + 1, j, k), u_identifer(i, j, k), u_identifer(i + 1, j, k), u_boundary(inlet_direction1, inlet_magnitude1, alpha, theta, j, k), u_boundary(inlet_direction1, inlet_magnitude1, alpha, theta, j, k), u_boundary(inlet_direction2, inlet_magnitude2, alpha, theta, j, k), u_boundary(inlet_direction2, inlet_magnitude2, alpha, theta, j, k), p_boundary());

			setupBoundaryVelocity(v(i, j, k), v(i, j + 1, k), pressure.index(i, j, k), pressure.index(i, j - 1, k), pressure.index(i, j + 1, k), v_liquid_phi(i, j, k), v_liquid_phi(i, j + 1, k), v_identifer(i, j, k), v_identifer(i, j + 1, k), v_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), v_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), v_boundary(inlet_direction2, inlet_magnitude2, alpha, theta), v_boundary(inlet_direction2, inlet_magnitude2, alpha, theta), p_boundary());

			setupBoundaryVelocity(w(i, j, k), w(i, j, k + 1), pressure.index(i, j, k), pressure.index(i, j, k - 1), pressure.index(i, j, k + 1), w_liquid_phi(i, j, k), w_liquid_phi(i, j, k + 1), w_identifer(i, j, k), w_identifer(i, j, k + 1), w_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), w_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), w_boundary(inlet_direction2, inlet_magnitude2, alpha, theta), w_boundary(inlet_direction2, inlet_magnitude2, alpha, theta), p_boundary());
		}
	}
}

#else
void Simple::solve_velocity_component(Array3d& vel, Array3d& vel_ss, Array3BT& identifier, Array3d& vel_phi, double dt, double alpha,double theta)
{
	int system_size = vel.ni * vel.nj * vel.nk;
	b.resize(system_size);
	A.resize(system_size, system_size);
	std::vector<TripletD> coefficients;
	Eigen::VectorXd x(system_size);

	b.setZero();

	double term = kinetic_coefficient * dt / (blood_density*dx*dx);
	auto setupNeighbors = [&](
		double& s_i,
		int i,
		int i_minus,
		int i_plus,
		double phi_i_minus,
		double phi_i_plus,
		CellType cell_i_minus,
		CellType cell_i_plus,
		double vb_1_plus,
		double vb_1_minus,
		double vb_2_plus,
		double vb_2_minus)
	{
		if (cell_i_minus == CellType::Inside && cell_i_plus == CellType::Inside)
		{
			coefficients.push_back(TripletD(i, i_minus, -term));
			coefficients.push_back(TripletD(i, i_plus, -term));
			coefficients.push_back(TripletD(i, i, 2 * term));
		}
		else if (cell_i_minus == CellType::Inside && cell_i_plus != CellType::Inside)
		{
			if (cell_i_plus == CellType::Static)
			{
				double A = phi_i_plus / (2 * dx_mm - phi_i_plus);
				coefficients.push_back(TripletD(i, i_minus, -term * (1-A)));
				coefficients.push_back(TripletD(i, i, 2 * term));
			}
			else if (cell_i_plus == CellType::Inlet1)
			{
				double A =  phi_i_plus / (2 * dx_mm - phi_i_plus);
				coefficients.push_back(TripletD(i, i_minus, -term * (1-A)));
				coefficients.push_back(TripletD(i, i, 2 * term));

				s_i += term * (1+A)*vb_1_plus;
			}
			else if (cell_i_plus == CellType::Inlet2)
			{
				double A = phi_i_plus / (2 * dx_mm - phi_i_plus);
				coefficients.push_back(TripletD(i, i_minus, -term * (1 - A)));
				coefficients.push_back(TripletD(i, i, 2 * term));

				s_i += term * (1 + A)*vb_2_plus;
			}
			else
			{
				coefficients.push_back(TripletD(i, i_minus, -term));
				coefficients.push_back(TripletD(i, i, term));
			}
		}
		else if (cell_i_minus != CellType::Inside && cell_i_plus == CellType::Inside)
		{
			if (cell_i_minus == CellType::Static)
			{
				double A =  phi_i_minus / (2 * dx_mm - phi_i_minus);
				coefficients.push_back(TripletD(i, i_plus, -term * (1-A)));
				coefficients.push_back(TripletD(i, i, 2 * term));
			}
			else if (cell_i_minus == CellType::Inlet1)
			{
				double A =  phi_i_minus / (2 * dx_mm - phi_i_minus);
				coefficients.push_back(TripletD(i, i_plus, -term * (1-A)));
				coefficients.push_back(TripletD(i, i, 2 * term));

				s_i += term * (1+A)*vb_1_minus;
			}
			else if (cell_i_minus == CellType::Inlet2)
			{
				double A = phi_i_minus / (2 * dx_mm - phi_i_minus);
				coefficients.push_back(TripletD(i, i_plus, -term * (1 - A)));
				coefficients.push_back(TripletD(i, i, 2 * term));

				s_i += term * (1 + A)*vb_2_minus;
			}
			else
			{
				coefficients.push_back(TripletD(i, i_plus, -term));
				coefficients.push_back(TripletD(i, i, term));
			}
		}
		else
		{
			//TODO: Evaluate how to handle this case

			//double theta = fraction_inside(phi_i, phi_j);
			//if (theta < 0.01f) theta = 0.01f;
			////coefficients.push_back(TripletD(i, j_opposite, d*(1 - theta) / (theta + 1)));
			//coefficients.push_back(TripletD(p_i, p_i, term / theta));
		}
	};

	for (int k = 0; k < vel.nk; ++k) {
		for (int j = 0; j < vel.nj; ++j) {
			for (int i = 0; i < vel.ni; ++i) {
				int index = vel.index(i, j, k);
				if (identifier(i, j, k) == CellType::Inside)
				{
					double& b_ijk = b(index);
					b(index) += vel_ss(i, j, k);
					coefficients.push_back(TripletD(index, index, 1.0));
					if (vel.ni > ni) 
					{
						setupNeighbors(b_ijk, index, vel.index(i - 1, j, k), vel.index(i + 1, j, k), vel_phi(i - 1, j, k), vel_phi(i + 1, j, k), identifier(i - 1, j, k), identifier(i + 1, j, k), u_boundary(inlet_direction1,inlet_magnitude1, alpha,theta,j,k), u_boundary(inlet_direction1, inlet_magnitude1, alpha, theta,j,k), u_boundary(inlet_direction2, inlet_magnitude2, alpha, theta,j,k), u_boundary(inlet_direction2, inlet_magnitude2, alpha, theta,j,k));

						setupNeighbors(b_ijk, index, vel.index(i, j - 1, k), vel.index(i, j + 1, k), vel_phi(i, j - 1, k), vel_phi(i, j + 1, k), identifier(i, j - 1, k), identifier(i, j + 1, k), u_boundary(inlet_direction1, inlet_magnitude1, alpha, theta,j+1,k), u_boundary(inlet_direction1, inlet_magnitude1, alpha, theta,j-1,k), u_boundary(inlet_direction2, inlet_magnitude2, alpha, theta,j+1,k), u_boundary(inlet_direction2, inlet_magnitude2, alpha, theta,j-1,k));

						setupNeighbors(b_ijk, index, vel.index(i, j, k - 1), vel.index(i, j, k + 1), vel_phi(i, j, k - 1), vel_phi(i, j, k + 1), identifier(i, j, k - 1), identifier(i, j, k + 1), u_boundary(inlet_direction1, inlet_magnitude1, alpha, theta,j,k+1), u_boundary(inlet_direction1, inlet_magnitude1, alpha, theta,j,k-1), u_boundary(inlet_direction2, inlet_magnitude2, alpha, theta,j,k+1), u_boundary(inlet_direction2, inlet_magnitude2, alpha, theta,j,k-1));
					}
					else if (vel.nj > nj)
					{
						setupNeighbors(b_ijk, index, vel.index(i - 1, j, k), vel.index(i + 1, j, k), vel_phi(i - 1, j, k), vel_phi(i + 1, j, k), identifier(i - 1, j, k), identifier(i + 1, j, k), v_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), v_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), v_boundary(inlet_direction2, inlet_magnitude2, alpha, theta),v_boundary(inlet_direction2, inlet_magnitude2, alpha, theta));

						setupNeighbors(b_ijk, index, vel.index(i, j - 1, k), vel.index(i, j + 1, k), vel_phi(i, j - 1, k), vel_phi(i, j + 1, k), identifier(i, j - 1, k), identifier(i, j + 1, k), v_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), v_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), v_boundary(inlet_direction2, inlet_magnitude2, alpha, theta),v_boundary(inlet_direction2, inlet_magnitude2, alpha, theta));

						setupNeighbors(b_ijk, index, vel.index(i, j, k - 1), vel.index(i, j, k + 1), vel_phi(i, j, k - 1), vel_phi(i, j, k + 1), identifier(i, j, k - 1), identifier(i, j, k + 1), v_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), v_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), v_boundary(inlet_direction2, inlet_magnitude2, alpha, theta),v_boundary(inlet_direction2, inlet_magnitude2, alpha, theta));
					}
					else if (vel.nk > nk)
					{
						setupNeighbors(b_ijk, index, vel.index(i - 1, j, k), vel.index(i + 1, j, k), vel_phi(i - 1, j, k), vel_phi(i + 1, j, k), identifier(i - 1, j, k), identifier(i + 1, j, k), w_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), w_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), w_boundary(inlet_direction2, inlet_magnitude2, alpha, theta),w_boundary(inlet_direction2, inlet_magnitude2, alpha, theta));

						setupNeighbors(b_ijk, index, vel.index(i, j - 1, k), vel.index(i, j + 1, k), vel_phi(i, j - 1, k), vel_phi(i, j + 1, k), identifier(i, j - 1, k), identifier(i, j + 1, k), w_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), w_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), w_boundary(inlet_direction2, inlet_magnitude2, alpha, theta),w_boundary(inlet_direction2, inlet_magnitude2, alpha, theta));

						setupNeighbors(b_ijk, index, vel.index(i, j, k - 1), vel.index(i, j, k + 1), vel_phi(i, j, k - 1), vel_phi(i, j, k + 1), identifier(i, j, k - 1), identifier(i, j, k + 1), w_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), w_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), w_boundary(inlet_direction2, inlet_magnitude2, alpha, theta),w_boundary(inlet_direction2, inlet_magnitude2, alpha, theta));
					}
				}
			}
		}
	}

	A.setFromTriplets(coefficients.begin(), coefficients.end());
	Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> so;
	so.compute(A);
	x = so.solve(b);
	std::cout << "#iterations:   " << so.iterations() << "    #estimated error:   " << so.error() << std::endl;

	//update velocity
	for (int k = 0; k < vel.nk; ++k) {
		for (int j = 0; j < vel.nj; ++j) {
			for (int i = 0; i < vel.ni; ++i) {
				int index = vel.index(i, j, k);
				if (identifier(i, j, k) == CellType::Inside)
					vel(i, j, k) = x(index);
			}
		}
	}
}

void Simple::solve_pressure(Array3d& pressure, Array3d& u_in, Array3d& v_in, Array3d& w_in, double dt, double alpha,double theta)
{
	int system_size = pressure.ni * pressure.nj * pressure.nk;
	b.resize(system_size);
	A.resize(system_size, system_size);
	std::vector<TripletD> coefficients;
	Eigen::VectorXd x(system_size);

	b.setZero();

	pressure.set_zero();

	double term = dx * blood_density / dt;
	auto setupPPE = [&](
		double& s_i,
		int p_i,
		int p_i_minus,
		int p_i_plus,
		double v_i_minus,
		double v_i_plus,
		double v_phi_i_minus,
		double v_phi_i_plus,
		CellType v_type_i_minus,
		CellType v_type_i_plus,
		double vb_1_plus,
		double vb_1_minus,
		double vb_2_plus,
		double vb_2_minus,
		double pb_1)
	{
		CellType p_type_i_minus = p_identifer[p_i_minus];
		CellType p_type_i_plus = p_identifer[p_i_plus];

		double p_phi_i_plus = liquid_phi[p_i_plus];
		double p_phi_i_minus = liquid_phi[p_i_minus];

		if (p_type_i_minus == CellType::Inside && p_type_i_plus == CellType::Inside)
		{
			coefficients.push_back(TripletD(p_i, p_i_plus, -1.0));
			coefficients.push_back(TripletD(p_i, p_i_minus, -1.0));
			coefficients.push_back(TripletD(p_i, p_i, 2.0));

			s_i += -term * (v_i_plus - v_i_minus);
		}
		else if (p_type_i_minus == CellType::Inside && p_type_i_plus != CellType::Inside)
		{
				//TODO: add contribution from boundary velocity
				if (p_type_i_plus == CellType::Static)
				{
					//Impose velocity boundary conditions
					double A = -dx_mm / (v_phi_i_plus - dx_mm);
					coefficients.push_back(TripletD(p_i, p_i_minus, -A));
					coefficients.push_back(TripletD(p_i, p_i, A));

					s_i += term * v_i_minus* A;
				}
				else if (p_type_i_plus == CellType::Inlet1)
				{
					//Impose velocity boundary conditions
					double A = -dx_mm / (v_phi_i_plus - dx_mm);
					coefficients.push_back(TripletD(p_i, p_i_minus, -A));
					coefficients.push_back(TripletD(p_i, p_i, A));

					s_i += term * v_i_minus*A;
					//TODO: compute the inlet velocity boundary
					s_i += -term * vb_1_plus * A;
				}
				else if (p_type_i_plus == CellType::Inlet2)
				{
					//Impose velocity boundary conditions
					double A = -dx_mm / (v_phi_i_plus - dx_mm);
					coefficients.push_back(TripletD(p_i, p_i_minus, -A));
					coefficients.push_back(TripletD(p_i, p_i, A));

					s_i += term * v_i_minus*A;
					//TODO: compute the inlet velocity boundary
					s_i += -term * vb_2_plus * A;
				}
				else if (p_type_i_plus == CellType::Outlet1)
				{
					//TODO: check the solvability
					if (v_type_i_plus == CellType::Inside)
					{
						//Impose pressure boundary conditions
						double A = p_phi_i_plus / (2 * dx_mm - p_phi_i_plus);

						coefficients.push_back(TripletD(p_i, p_i_minus, -(1 - A)));
						coefficients.push_back(TripletD(p_i, p_i, 2.0));

						//TODO: check the sign for this term
						s_i += -term * (v_i_plus - v_i_minus);

						//TODO: check the sign for this term
						s_i += (1 + A)*pb_1;
					}
				}
		}
		else if (p_type_i_minus != CellType::Inside && p_type_i_plus == CellType::Inside)
		{
			if (p_type_i_minus == CellType::Static)
			{
			    //Impose velocity boundary conditions
				double A = -dx_mm / (v_phi_i_minus - dx_mm);

				coefficients.push_back(TripletD(p_i, p_i_plus, -A));
				coefficients.push_back(TripletD(p_i, p_i, A));

				s_i += -term * v_i_plus* A;
			}
			else if (p_type_i_minus == CellType::Inlet1)
			{
				//Impose velocity boundary conditions
				double A = -dx_mm / (v_phi_i_minus - dx_mm);

				coefficients.push_back(TripletD(p_i, p_i_plus, -A));
				coefficients.push_back(TripletD(p_i, p_i,A));

				s_i += -term * v_i_plus*A;

				s_i += term * vb_1_minus *  A;
			}
			else if (p_type_i_minus == CellType::Inlet2)
			{
				//Impose velocity boundary conditions
				double A = -dx_mm / (v_phi_i_minus - dx_mm);

				coefficients.push_back(TripletD(p_i, p_i_plus, -A));
				coefficients.push_back(TripletD(p_i, p_i, A));

				s_i += -term * v_i_plus*A;

				s_i += term * vb_2_minus *  A;
			}
			else if (p_type_i_minus == CellType::Outlet1)
			{
				//TODO: check the solvability
				if (v_type_i_minus == CellType::Inside)
				{
					//Impose pressure boundary conditions
					double A = p_phi_i_minus / (2 * dx_mm - p_phi_i_minus);
					coefficients.push_back(TripletD(p_i, p_i_plus, -(1 - A)));
					coefficients.push_back(TripletD(p_i, p_i, 2.0));

					//TODO: check the sign for this term
					s_i += -term * (v_i_plus - v_i_minus);

					//TODO: check the sign for this term
					s_i += (1 + A)*pb_1;
				}
			}
		}
		else
		{
			//TODO: Evaluate how to handle this case

// 			double theta = fraction_inside(phi_i, phi_j);
// 			if (theta < 0.01f) theta = 0.01f;
// 			//coefficients.push_back(TripletD(i, j_opposite, d*(1 - theta) / (theta + 1)));
// 			coefficients.push_back(TripletD(p_i, p_i, term / theta));
		}
	};

	for (int k = 0; k <nk; ++k) {
		for (int j = 0; j < nj; ++j) {
			for (int i = 0; i < ni; ++i) {
				int index = pressure.index(i, j, k);
				if (p_identifer(i,j,k) == CellType::Inside)
				{
					double source_ijk = 0.0;
					setupPPE(source_ijk, index, pressure.index(i - 1, j, k), pressure.index(i + 1, j, k),u_in(i, j, k),u_in(i + 1, j, k),u_liquid_phi(i, j, k),u_liquid_phi(i + 1, j, k),u_identifer(i, j, k),u_identifer(i + 1, j, k), u_boundary(inlet_direction1,inlet_magnitude1,alpha, theta,j,k), u_boundary(inlet_direction1, inlet_magnitude1, alpha, theta,j,k), u_boundary(inlet_direction2, inlet_magnitude2, alpha, theta,j,k), u_boundary(inlet_direction2, inlet_magnitude2, alpha, theta,j,k),p_boundary());

					setupPPE(source_ijk,index,pressure.index(i, j - 1, k),pressure.index(i, j + 1, k),v_in(i, j, k),v_in(i, j + 1, k),v_liquid_phi(i, j, k),v_liquid_phi(i, j + 1, k),v_identifer(i, j, k),v_identifer(i, j + 1, k), v_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), v_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), v_boundary(inlet_direction2, inlet_magnitude2, alpha, theta), v_boundary(inlet_direction2, inlet_magnitude2, alpha, theta), p_boundary());

					setupPPE(source_ijk,index,pressure.index(i, j, k - 1),pressure.index(i, j, k + 1),w_in(i, j, k),w_in(i, j, k + 1),w_liquid_phi(i, j, k),w_liquid_phi(i, j, k + 1),w_identifer(i, j, k),w_identifer(i, j, k + 1), w_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), w_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), w_boundary(inlet_direction2, inlet_magnitude2, alpha, theta), w_boundary(inlet_direction2, inlet_magnitude2, alpha, theta), p_boundary());

					b[index] = source_ijk;
				}
			}
		}
	}
	A.setFromTriplets(coefficients.begin(), coefficients.end());
	Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> so;
	so.compute(A);
	x = so.solve(b);
	std::cout << "Pressure iterations:   " << so.iterations() << "    #estimated error:   " << so.error() << std::endl;

	//update velocity
	for (int k = 0; k < nk; ++k) {
		for (int j = 0; j < nj; ++j) {
			for (int i = 0; i < ni; ++i) {
				int index = pressure.index(i, j, k);
				if (p_identifer(i, j, k) == CellType::Inside)
					pressure(i, j, k) = x(index);
			}
		}
	}
 	convergence_indicator = b.norm();
}

void Simple::update_velocity(Array3d& u, Array3d& v, Array3d& w, Array3d& pressure, double dt, double alpha,double theta)
{
	std::cout << "************this conditation is UNBLOCK" << std::endl;
	double term = dt / (dx * blood_density);

	for (int k = 0; k < u.nk; ++k) for (int j = 0; j < u.nj; ++j) for (int i = 1; i < u.ni - 1; ++i)
	{
		if (u_identifer(i, j, k) == CellType::Inside)
		{
			if (p_identifer(i - 1, j, k) == CellType::Inside && p_identifer(i, j, k) == CellType::Inside)
				u(i, j, k) -= term * (pressure(i, j, k) - pressure(i - 1, j, k));
		}
	}
	for (int k = 0; k < v.nk; ++k) for (int j = 1; j < v.nj - 1; ++j) for (int i = 0; i < v.ni; ++i)
	{
		if (v_identifer(i, j, k) == CellType::Inside)
		{
			if (p_identifer(i, j - 1, k) == CellType::Inside && p_identifer(i, j, k) == CellType::Inside)
				v(i, j, k) -= term * (pressure(i, j, k) - pressure(i, j - 1, k));
		}
	}
	for (int k = 1; k < w.nk - 1; ++k) for (int j = 0; j < w.nj; ++j) for (int i = 0; i < w.ni; ++i)
	{
		if (w_identifer(i, j, k) == CellType::Inside)
		{
			if (p_identifer(i, j, k - 1) == CellType::Inside && p_identifer(i, j, k) == CellType::Inside)
				w(i, j, k) -= term * (pressure(i, j, k) - pressure(i, j, k - 1));
		}
	}

	auto setupBoundaryVelocity = [&](
		double& v_i_minus,
		double& v_i_plus,
		int p_i,
		int p_i_minus,
		int p_i_plus,
		double v_phi_i_minus,
		double v_phi_i_plus,
		CellType v_type_i_minus,
		CellType v_type_i_plus,
		double vb_1_plus,
		double vb_1_minus,
		double vb_2_plus,
		double vb_2_minus,
		double pb_1)
	{
		CellType p_type_i_minus = p_identifer[p_i_minus];
		CellType p_type_i_plus = p_identifer[p_i_plus];

		double p_phi_i_minus = liquid_phi[p_i_minus];
		double p_phi_i_plus = liquid_phi[p_i_plus];

		if (p_type_i_minus == CellType::Inside && p_type_i_plus != CellType::Inside)
		{
			if (v_type_i_plus == CellType::Inside)
			{
				//TODO: add contribution from boundary velocity
				if (p_type_i_plus == CellType::Static)
				{
					double A = -dx_mm / (v_phi_i_plus - dx_mm);

					v_i_plus = (1-A) * v_i_minus;
				}
				else if (p_type_i_plus == CellType::Inlet1)
				{
					double A = -dx_mm / (v_phi_i_plus - dx_mm);

					v_i_plus = (1 - A) * v_i_minus + A * vb_1_plus;
				}
				else if (p_type_i_plus == CellType::Inlet2)
				{
					double A = -dx_mm / (v_phi_i_plus - dx_mm);

					v_i_plus = (1 - A) * v_i_minus + A * vb_2_plus;
				}
				else if (p_type_i_plus == CellType::Outlet1)
				{
					double A = p_phi_i_plus / (2 * dx_mm - p_phi_i_plus);

					double pressure_i_plus_new = -A * pressure[p_i_minus] + (1 + A)*pb_1;
					v_i_plus -= term * (pressure_i_plus_new - pressure[p_i]);
				}
				else if (p_type_i_plus == CellType::Outlet2)
				{
					v_i_plus = v_i_minus;
				}
			}			
		}
		else if (p_type_i_minus != CellType::Inside && p_type_i_plus == CellType::Inside)
		{
			if (v_type_i_minus == CellType::Inside)
			{
				if (p_type_i_minus == CellType::Static)
				{
					double A = -dx_mm / (v_phi_i_minus - dx_mm);

					v_i_minus = (1-A) * v_i_plus;
				}
				else if (p_type_i_minus == CellType::Inlet1)
				{
					double A = -dx_mm / (v_phi_i_minus - dx_mm);

					v_i_minus = (1 - A) * v_i_plus + A * vb_1_minus;
				}
				else if (p_type_i_minus == CellType::Inlet2)
				{
					double A = -dx_mm / (v_phi_i_minus - dx_mm);

					v_i_minus = (1 - A) * v_i_plus + A * vb_2_minus;
				}
				else if (p_type_i_minus == CellType::Outlet1)
				{
					double A = p_phi_i_minus / (2 * dx_mm - p_phi_i_minus);

					double pressure_i_minus_new = -A * pressure[p_i_plus] + (1 + A)*pb_1;
					v_i_minus -= term * (pressure[p_i] - pressure_i_minus_new);
				}
				else if (p_type_i_minus == CellType::Outlet2)
				{
					v_i_minus = v_i_plus;
				}
			}
		}
	};
	//Impose boundary conditions
	for (int k = 0; k < pressure.nk; ++k) for (int j = 0; j < pressure.nj; ++j) for (int i = 0; i < pressure.ni; ++i)
	{
		if (p_identifer(i, j, k) == CellType::Inside)
		{
			setupBoundaryVelocity(u(i, j, k), u(i + 1, j, k), pressure.index(i, j, k), pressure.index(i - 1, j, k), pressure.index(i + 1, j, k), u_liquid_phi(i, j, k), u_liquid_phi(i + 1, j, k), u_identifer(i, j, k), u_identifer(i + 1, j, k), u_boundary(inlet_direction1, inlet_magnitude1, alpha, theta,j,k), u_boundary(inlet_direction1, inlet_magnitude1, alpha, theta,j,k), u_boundary(inlet_direction2, inlet_magnitude2, alpha, theta,j,k), u_boundary(inlet_direction2, inlet_magnitude2, alpha, theta,j,k), p_boundary());

			setupBoundaryVelocity(v(i, j, k), v(i, j + 1, k), pressure.index(i, j, k), pressure.index(i, j - 1, k), pressure.index(i, j + 1, k), v_liquid_phi(i, j, k), v_liquid_phi(i, j + 1, k), v_identifer(i, j, k), v_identifer(i, j + 1, k), v_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), v_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), v_boundary(inlet_direction2, inlet_magnitude2, alpha, theta), v_boundary(inlet_direction2, inlet_magnitude2, alpha, theta), p_boundary());

			setupBoundaryVelocity(w(i, j, k), w(i, j, k + 1), pressure.index(i, j, k), pressure.index(i, j, k - 1), pressure.index(i, j, k + 1), w_liquid_phi(i, j, k), w_liquid_phi(i, j, k + 1), w_identifer(i, j, k), w_identifer(i, j, k + 1), w_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), w_boundary(inlet_direction1, inlet_magnitude1, alpha, theta), w_boundary(inlet_direction2, inlet_magnitude2, alpha, theta), w_boundary(inlet_direction2, inlet_magnitude2, alpha, theta), p_boundary());
		}
	}
}
#endif // BLOCK

void Simple::classifyBoundary(CellType& ct, Vec3d pos, mfd::DistanceField3D& inletSDF1, mfd::DistanceField3D& inletSDF2, mfd::DistanceField3D& outletSDF1, mfd::DistanceField3D& outletSDF2)
{
	double d_inlet1, d_inlet2, d_outlet1, d_outlet2;
	inletSDF1.GetDistance(pos, d_inlet1);
	inletSDF2.GetDistance(pos, d_inlet2);
	outletSDF1.GetDistance(pos, d_outlet1);
	outletSDF2.GetDistance(pos, d_outlet2);
	if (d_inlet1 < 0)
	{
		ct = CellType::Inlet1;
	}
	else if (d_inlet2 < 0)
	{
		ct = CellType::Inlet2;
	}
	else if (d_outlet1 < 0)
	{
		ct = CellType::Outlet1;
	}
	else if (d_outlet2 < 0)
	{
		ct = CellType::Outlet2;
	}
	else
	{
		ct = CellType::Static;
	}
}

double Simple::u_boundary(Vec3d& in_dir,double& in_mag, double al,double the,int grid_j,int grid_k)
{
	//double location_j = lower_vertex[1] + dx_mm * (grid_j + 0.5);
	//double location_k = lower_vertex[2] + dx_mm * (grid_k + 0.5);
	//in_mag=0.4 * (1.0 - ((sqr(location_j) + sqr(location_k)) / sqr(5.4)));
	Vec3d inlet_vel = -in_mag * in_dir;

	return inlet_vel[0];

}

double Simple::v_boundary(Vec3d& in_dir, double& in_mag, double al,double the)
{
	Vec3d inlet_vel = -in_mag * in_dir;
	return inlet_vel[1];
}

double Simple::w_boundary(Vec3d& in_dir, double& in_mag, double al, double the)
{
	Vec3d inlet_vel = -in_mag * in_dir;
	return inlet_vel[2];
}

double Simple::p_boundary()
{
	return 0.0;
}

double Simple::cfl()
{
	double maxvel = 0.0,cfl_v=1.0;
	for (unsigned int i = 0; i < u.a.size(); ++i)
		maxvel = max(maxvel, abs(u.a[i]));
	for (unsigned int i = 0; i < v.a.size(); ++i)
		maxvel = max(maxvel, abs(v.a[i]));
	for (unsigned int i = 0; i < w.a.size(); ++i)
		maxvel = max(maxvel, abs(w.a[i]));

	std::printf("max velocity: %f \n", maxvel);

	if (maxvel > 0)
		cfl_v = dx / maxvel;

	return cfl_v;
}

void Simple::compute_velocity(double dt)
{
	int cycle = floor(dt / 0.8);
	dt -= 0.8 * cycle;
	inlet_magnitude1 = 0.5 + 0.25 * sin(2.5 * dt * M_PI - 0.5 * M_PI);
}

void Simple::export_status(std::string path, int frame)
{
	std::stringstream strout;
	strout << path << "NORMAL1c_BLOCK.txt";
	string filepath = strout.str();
	cout << "the writed file is: " << filepath << endl;
	std::ofstream status_out(filepath.c_str());
	if (!status_out.good())
	{
		std::printf("Failed to open status!\n");
		return;
	}

	status_out << (unsigned int)ni << " " << (unsigned int)nj << " " << (unsigned int)nk << std::endl;

	double zero_value = 0.0;
	for (int k = 0; k < nk; ++k) for (int j = 0; j < nj; ++j) for (int i = 0; i < ni; ++i)
	{
		if (p_identifer(i, j, k) == CellType::Inside)
			status_out << p(i, j, k) << std::endl;
		else
			status_out << zero_value << std::endl;
	}

	for (int k = 0; k < nk; ++k) for (int j = 0; j < nj; ++j) for (int i = 0; i < ni + 1; ++i)
	{
		if (u_identifer(i, j, k) == CellType::Inside)
			status_out << u(i, j, k) << std::endl;
		else
			status_out << zero_value << std::endl;
	}

	for (int k = 0; k < nk; ++k) for (int j = 0; j < nj + 1; ++j) for (int i = 0; i < ni; ++i)
	{
		if (v_identifer(i, j, k) == CellType::Inside)
			status_out << v(i, j, k) << std::endl;
		else
			status_out << zero_value << std::endl;
	}

	for (int k = 0; k < nk + 1; ++k) for (int j = 0; j < nj; ++j) for (int i = 0; i < ni; ++i)
	{
		if (w_identifer(i, j, k) == CellType::Inside)
			status_out << w(i, j, k) << std::endl;
		else
			status_out << zero_value << std::endl;
	}

	std::cout << "Writing to: " << filepath << std::endl;
	status_out.close();
}

void Simple::export_velocity_vtk(std::string path)
{
	std::stringstream strout;
	strout << path << "NORMAL1.vtk";
	string filepath = strout.str();
	cout << "the writed file is: " << filepath << endl;

	vtkSmartPointer<vtkPolyData> data = vtkSmartPointer<vtkPolyData>::New();
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkCellArray> vertices = vtkSmartPointer<vtkCellArray>::New();
	vtkSmartPointer<vtkDoubleArray> velocity = vtkSmartPointer<vtkDoubleArray>::New();
	velocity->SetNumberOfComponents(3);
	//velocity->SetNumberOfTuples(ni*nj*nk);
	unsigned int index = 0;
	for (int k = 0; k < nk; ++k) for (int j = 0; j < nj; ++j) for (int i = 0; i < ni; ++i)
	{
		if (p_identifer(i, j, k) == CellType::Inside)
		{
			Vec3d pos(lower_vertex[0] + (i + 0.5) * dx_mm, lower_vertex[1] + (j + 0.5) * dx_mm, lower_vertex[2] + (k + 0.5) * dx_mm);
			double vertex_u = interpolate_value(Vec3d(i + 0.5, j, k), u);
			double vertex_v = interpolate_value(Vec3d(i, j + 0.5, k), v);
			double vertex_w = interpolate_value(Vec3d(i, j, k + 0.5), w);

			vtkIdType pid[1];
			pid[0] = points->InsertNextPoint(pos[0], pos[1], pos[2]);
			velocity->InsertTuple3(index, vertex_u, vertex_v, vertex_w);
			vertices->InsertNextCell(1, pid);
			index++;
		}
	}

	data->SetPoints(points);
	data->SetVerts(vertices);
	data->GetPointData()->SetVectors(velocity);

	vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
	writer->SetFileName(filepath.c_str());
	writer->SetInputData(data);
	writer->Write();
}

void Simple::export_pressure_vtk(std::string path)
{
	std::stringstream strout;
	strout << path << "NORMAL1_p.vtk";
	string filepath = strout.str();
	cout << "the writed file is: " << filepath << endl;

	vtkSmartPointer<vtkPolyData> data = vtkSmartPointer<vtkPolyData>::New();
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkCellArray> vertices = vtkSmartPointer<vtkCellArray>::New();
	vtkSmartPointer<vtkDoubleArray> vertex_pressure = vtkSmartPointer<vtkDoubleArray>::New();
	//velocity->SetNumberOfComponents(1);
	//velocity->SetNumberOfTuples(ni*nj*nk);
	unsigned int index = 0;
	for (int k = 0; k < nk; ++k) for (int j = 0; j < nj; ++j) for (int i = 0; i < ni; ++i)
	{
		if (p_identifer(i, j, k) == CellType::Inside)
		{
			Vec3d pos(lower_vertex[0] + (i + 0.5) * dx_mm, lower_vertex[1] + (j + 0.5) * dx_mm, lower_vertex[2] + (k + 0.5) * dx_mm);

			vtkIdType pid[1];
			pid[0] = points->InsertNextPoint(pos[0], pos[1], pos[2]);
			vertex_pressure->InsertTuple1(index,p(i,j,k));
			vertices->InsertNextCell(1, pid);
			index++;
		}
	}

	data->SetPoints(points);
	data->SetVerts(vertices);
	data->GetPointData()->SetScalars(vertex_pressure);

	vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
	writer->SetFileName(filepath.c_str());
	writer->SetInputData(data);
	writer->Write();
}

bool Simple::import_status(std::string path)
{
	std::ifstream status_in(path);
	if (!status_in.good())
	{
		printf("Failed to open files!\n");
		return false;
	}

	int in_i, in_j, in_k;
	status_in >> in_i >> in_j >> in_k;
	cout << in_i << " " << in_j << " " << in_k << endl;
	if ((in_i != ni || in_j != nj) || in_k != nk)
	{
		std::cout << "the fill is not match with the initialized model!" << std::endl;
		return false;
	}


	for (int k = 0; k < in_k; ++k) for (int j = 0; j < in_j; ++j) for (int i = 0; i < in_i; ++i) 
	{
		double pre;
		status_in >> pre;
		p(i, j, k) = pre;
	}
	for (int k = 0; k < in_k; ++k) for (int j = 0; j < in_j; ++j) for (int i = 0; i < in_i + 1; ++i)
	{
		double vel;
		status_in >> vel;
		u(i, j, k) = vel;
	}

	for (int k = 0; k < in_k; ++k) for (int j = 0; j < in_j + 1; ++j) for (int i = 0; i <in_i; ++i) 
	{
		double vel;
		status_in >> vel;
		v(i, j, k) = vel;
	}

	for (int k = 0; k < in_k + 1; ++k) for (int j = 0; j < in_j; ++j) for (int i = 0; i < in_i; ++i) 
	{
		double vel;
		status_in >> vel;
		w(i, j, k) = vel;
	}
	std::cout << "import status successfully. " << std::endl;
	return true;
}

void Simple::extrapolate(Array3d& grid, Array3BT iden) 
{
	Array3d temp_grid = grid;
	Array3BT old_iden(iden.ni, iden.nj, iden.nk);
	for (int layers = 0; layers < 10; ++layers) {
		old_iden = iden;
		for (int k = 1; k < grid.nk - 1; ++k) for (int j = 1; j < grid.nj - 1; ++j) for (int i = 1; i < grid.ni - 1; ++i) {
			double sum = 0;
			int count = 0;

			if (old_iden(i, j, k)!=CellType::Inside) {

				if (old_iden(i + 1, j, k)==CellType::Inside) {
					sum += grid(i + 1, j, k);
					++count;
				}
				if (old_iden(i - 1, j, k)==CellType::Inside) {
					sum += grid(i - 1, j, k);
					++count;
				}
				if (old_iden(i, j + 1, k) == CellType::Inside) {
					sum += grid(i, j + 1, k);
					++count;
				}
				if (old_iden(i, j - 1, k) == CellType::Inside) {
					sum += grid(i, j - 1, k);
					++count;
				}
				if (old_iden(i, j, k + 1) == CellType::Inside) {
					sum += grid(i, j, k + 1);
					++count;
				}
				if (old_iden(i, j, k - 1) == CellType::Inside) {
					sum += grid(i, j, k - 1);
					++count;
				}
				//If any of neighbour cells were valid, 
				//assign the cell their average value and tag it as valid
				if (count > 0) {
					temp_grid(i, j, k) = sum / (double)count;
					iden(i, j, k) = CellType::Inside;
				}
			}
		}
		grid = temp_grid;
	}
}

//statistic 3D pipe
void Simple::statistics(std::string face1, std::string face2, double alpha, double theta)
{
	////statistics velocity u
	//Vec3d u_orientation(cos(alpha)*cos(theta), sin(theta), -sin(alpha)*cos(theta));
	//statistics_velocity(u_identifer, u, u_orientation, alpha, theta);
	////statistics velocity v
	//Vec3d v_orientation(-cos(alpha)*sin(theta), cos(theta), sin(alpha)*sin(theta));
	//statistics_velocity(v_identifer, v, v_orientation, alpha, theta);
	////statistics velocity w
	//Vec3d w_orientation(sin(alpha), 0, cos(alpha));
	//statistics_velocity(w_identifer, w, w_orientation, alpha, theta);
	//statistics pressure
	double p_f1 = 0, p_f2 = 0;
	p_f1 = statistics_pressure(face1);
	p_f2 = statistics_pressure(face2);
	//std::cout << "the two face is: " << face1 << std::endl;
	//std::cout << "                 " << face2 << std::endl;
	std::cout << "the average pressure of face1 and face2 is: " << p_f1 << " " << p_f2 << " " << p_f1 - p_f2 << std::endl;
	std::cout << std::endl << std::endl;
}

double Simple::statistics_pressure(std::string face)
{
	//statistics pressure
	std::vector<Vec3d> vertexlist;
	std::string line;
	std::ifstream infile(face);
	if (!infile)
	{
		std::cerr << "Failed to open. Terminating.\n";
		exit(-1);
	}
	while (!infile.eof())
	{
		std::getline(infile, line);
		//.obj files sometimes contain vertex normals indicated by "vn"
		if (line.substr(0, 1) == std::string("v") && line.substr(0, 2) != std::string("vn"))
		{
			std::stringstream data(line);
			char c;
			Vec3d point;
			data >> c >> point[0] >> point[1] >> point[2];
			vertexlist.push_back(point);
		}
	}
	std::cout << "Read in " << vertexlist.size() << " vertices. " << std::endl;

	int num_p = 0;
	double total_p = 0;
	for (int i = 0; i < vertexlist.size(); i++)
	{
		Vec3d vertex = (vertexlist[i]-lower_vertex)/dx_mm- Vec3d(0.5, 0.5, 0.5);
		double vertex_liquid = interpolate_value(vertex, liquid_phi);
		if (vertex_liquid < 0)
		{
			double vertex_pressure = interpolate_value(vertex, p);
			total_p += vertex_pressure;
			num_p++;
		}
	}
	if (num_p > 0)
		return total_p / num_p;
	else
		return 0;
}

void Simple::statistics_velocity(Array3BT& identifer, Array3d& velocity, Vec3d& orientation, double alpha, double theta)
{
	cout << velocity.ni << " " << velocity.nj << " " << velocity.nk << endl;
	Array3d velocity_error;
	velocity_error.resize(velocity.ni, velocity.nj, velocity.nk);
	velocity_error.set_zero();

	//statistics velocity
	int number = 0;
	double total_error = 0.0, error_avg = 0.0, error_deviation = 0.0;
	double max_error = FLT_MIN, min_error = FLT_MAX;
	for (int k = 0; k < velocity.nk; k++) {
		for (int j = 0; j < velocity.nj; j++) {
			for (int i = 0; i < velocity.ni; i++) {
				if (identifer(i, j, k) == CellType::Inside)
				{
					Vec3d pnew = lower_vertex + Vec3d((i + ((ni + 1) - (velocity.ni))*0.5) * dx_mm, (j + ((nj + 1) - (velocity.nj))*0.5) * dx_mm, (k + ((nk + 1) - (velocity.nk))*0.5) * dx_mm);
					double f1 = pnew[0] * cos(alpha)*cos(theta) - pnew[1] * cos(alpha)*sin(theta) + pnew[2] * sin(alpha) + 8;
					double f2 = pnew[0] * cos(alpha)*cos(theta) - pnew[1] * cos(alpha)*sin(theta) + pnew[2] * sin(alpha) - 13;

					if (f1 > 0 && f2 < 0)
					{
						Vec3d o1(cos(alpha)*cos(theta), -cos(alpha)*sin(theta), sin(alpha));
						Vec3d o2(sin(theta), cos(theta), 0);
						Vec3d o3(-sin(alpha)*cos(theta), sin(alpha)*sin(theta), cos(alpha));

						Vec3d pold(dot(pnew, o1), dot(pnew, o2), dot(pnew, o3));

						double c = 0.4*(1.0 - ((sqr(pold[1]) + sqr(pold[2])) / sqr(5.4)));


						double vel = dot(Vec3d(c, 0, 0), orientation);
						velocity_error(i, j, k) =abs(vel - velocity(i, j, k)) / 0.2;

						max_error = max(max_error, velocity_error(i, j, k));
						min_error = min(min_error, velocity_error(i, j, k));
						total_error += abs(velocity_error(i, j, k));
						number++;
						//std::cout << "this grid is: " << i << " " << j << " " << k << " " << vel << " " << velocity(i, j, k) << " " << velocity_error(i, j, k) << endl;
					}
				}
			}
		}
	}
	error_avg = total_error / number;
	for (int k = 0; k < velocity.nk; k++) {
		for (int j = 0; j < velocity.nj; j++) {
			for (int i = 0; i < velocity.ni; i++) {
				if (identifer(i, j, k) == CellType::Inside)
				{
					Vec3d pnew = lower_vertex + Vec3d((i + ((ni + 1) - (velocity.ni))*0.5) * dx_mm, (j + ((nj + 1) - (velocity.nj))*0.5) * dx_mm, (k + ((nk + 1) - (velocity.nk))*0.5) * dx_mm);
					double f1 = pnew[0] * cos(alpha)*cos(theta) - pnew[1] * cos(alpha)*sin(theta) + pnew[2] * sin(alpha) + 10;
					double f2 = pnew[0] * cos(alpha)*cos(theta) - pnew[1] * cos(alpha)*sin(theta) + pnew[2] * sin(alpha) - 10;

					if (f1 > 0 && f2 < 0)
					{
						error_deviation += sqr(abs(velocity_error(i, j, k)) - error_avg);
					}
				}
			}
		}
	}
	std::cout << "the max and min error of u velocity is:     " << max_error << "  " << min_error << endl;
	std::cout << "the average error of u velocity is:         " << error_avg << endl;
	std::cout << "the deviation of error of u velocity is:    " << error_deviation / number << endl;
	std::cout << std::endl;
}

void Simple::face_vertex_pressure(std::string path1, std::string path2, double zero1, double zero2, std::string model_sim, std::string pout_path)
{
	Array3d pre1, pre2;
	pre1.resize(ni, nj, nk); pre2.resize(ni, nj, nk);
	pre1.set_zero(); pre2.set_zero();

	std::ifstream status_in1(path1), status_in2(path2);
	if ((!status_in1.good()) || (!status_in2.good()))
		std::cout << "Failed to open files!\n" << std::endl;

	//read unblocked data
	int in_i1, in_j1, in_k1;
	status_in1 >> in_i1 >> in_j1 >> in_k1;
	if ((in_i1 != ni || in_j1 != nj) || in_k1 != nk)
		std::cout << "the unblocked data fill is not match with the initialized model!" << std::endl;

	for (int k = 0; k < nk; ++k) for (int j = 0; j < nj; ++j) for (int i = 0; i < ni; ++i)
	{
		double pre;
		status_in1 >> pre;
		pre1(i, j, k) = pre;
	}
	std::cout << "import unblocked data successfully. " << std::endl;


	//read blocked data
	int in_i2, in_j2, in_k2;
	status_in2 >> in_i2 >> in_j2 >> in_k2;
	if ((in_i2 != ni || in_j2 != nj) || in_k2 != nk)
		std::cout << "the blocked data fill is not match with the initialized model!" << std::endl;

	for (int k = 0; k < nk; ++k) for (int j = 0; j < nj; ++j) for (int i = 0; i < ni; ++i)
	{
		double pre;
		status_in2 >> pre;
		pre2(i, j, k) = pre;
	}
	std::cout << "import blocked data successfully. " << std::endl;



	//compute pressure_diff
	for (int k = 0; k < nk; ++k) for (int j = 0; j < nj; ++j) for (int i = 0; i < ni; ++i)
		p(i, j, k) = (pre2(i, j, k) - zero2) - (pre1(i, j, k) - zero1);

	//extrapolate p
	Array3BT p_id_plus = p_identifer;
	extrapolate(p, p_id_plus);



	//read vertexs of obj model
	std::vector<Vec3d> vertexlist;
	std::string line;
	std::ifstream infile(model_sim);
	if (!infile)
	{
		std::cerr << "Failed to open. Terminating.\n";
		exit(-1);
	}
	while (!infile.eof())
	{
		std::getline(infile, line);
		//.obj files sometimes contain vertex normals indicated by "vn"
		if (line.substr(0, 1) == std::string("v") && line.substr(0, 2) != std::string("vn"))
		{
			std::stringstream data(line);
			char c;
			Vec3d point;
			data >> c >> point[0] >> point[1] >> point[2];
			vertexlist.push_back(point);
		}
	}
	std::cout << "Read in " << vertexlist.size() << " vertices. " << std::endl;



	//pressure interpolation
	std::vector<double> vertex_pressure;
	for (int i = 0; i < vertexlist.size(); i++)
	{
		Vec3d vertex = (vertexlist[i] - lower_vertex) / dx_mm - Vec3d(0.5, 0.5, 0.5);
		double vertex_p = interpolate_value(vertex, p);
		vertex_pressure.push_back(vertex_p);
	}



	//export face vertex pressure
	std::stringstream strout;
	strout << pout_path;
	string filepath = strout.str();
	std::ofstream status_out(filepath.c_str());
	if (!status_out.good())
	{
		std::printf("Failed to open status!\n");
		return;
	}

	status_out << vertexlist.size() << std::endl;
	for (int i = 0; i < vertexlist.size(); i++)
	{
		status_out << vertex_pressure[i] << std::endl;
	}
	std::cout << "Writing to: " << filepath << std::endl;
	status_out.close();
}

//compute velocity of one face
Vec3d Simple::statistics_vel(std::string face)
{
	//statistics velocity
	Vec3d face_v;
	std::vector<Vec3d> vertexlist;
	std::string line;
	std::ifstream infile(face);
	if (!infile)
	{
		std::cerr << "Failed to open. Terminating.\n";
		exit(-1);
	}
	while (!infile.eof())
	{
		std::getline(infile, line);
		//.obj files sometimes contain vertex normals indicated by "vn"
		if (line.substr(0, 1) == std::string("v") && line.substr(0, 2) != std::string("vn"))
		{
			std::stringstream data(line);
			char c;
			Vec3d point;
			data >> c >> point[0] >> point[1] >> point[2];
			vertexlist.push_back(point);
		}
	}
	std::cout << "Read in " << vertexlist.size() << " vertices. " << std::endl;

	int num_u = 0, num_v = 0, num_w = 0;
	double total_u = 0, total_v = 0, total_w = 0;
	//velocity u
	for (int i = 0; i < vertexlist.size(); i++)
	{
		Vec3d vertex = (vertexlist[i] - lower_vertex) / dx_mm - Vec3d(0, 0.5, 0.5);
		double vertex_liquid = interpolate_value(vertex, u_liquid_phi);
		if (vertex_liquid < 0)
		{
			double vertex_u = interpolate_value(vertex, u);
			total_u += vertex_u;
			num_u++;
		}
	}
	if (num_u > 0)
		face_v[0]=total_u/num_u;
	else
		face_v[0] = 0;

	//velocity v
	for (int i = 0; i < vertexlist.size(); i++)
	{
		Vec3d vertex = (vertexlist[i] - lower_vertex) / dx_mm - Vec3d(0.5, 0, 0.5);
		double vertex_liquid = interpolate_value(vertex, v_liquid_phi);
		if (vertex_liquid < 0)
		{
			double vertex_v = interpolate_value(vertex, v);
			total_v += vertex_v;
			num_v++;
		}
	}
	if (num_v > 0)
		face_v[1] = total_v / num_v;
	else
		face_v[1] = 0;

	//velocity w
	for (int i = 0; i < vertexlist.size(); i++)
	{
		Vec3d vertex = (vertexlist[i] - lower_vertex) / dx_mm - Vec3d(0.5, 0.5, 0);
		double vertex_liquid = interpolate_value(vertex, w_liquid_phi);
		if (vertex_liquid < 0)
		{
			double vertex_w = interpolate_value(vertex, w);
			total_w += vertex_w;
			num_w++;
		}
	}
	if (num_w > 0)
		face_v[2] = total_w / num_w;
	else
		face_v[2] = 0;

	return face_v;
}
//statistic 2D pipe
void Simple::statistics(double al)
{
	Array3d u_error, v_error, w_error;
	u_error.resize(ni + 1, nj, nk); v_error.resize(ni, nj + 1, nk); w_error.resize(ni, nj, nk + 1);
	u_error.set_zero();	v_error.set_zero();	w_error.set_zero();
	u_init.resize(ni + 1, nj, nk); v_init.resize(ni, nj + 1, nk); w_init.resize(ni, nj, nk + 1);
	u_init.set_zero(); v_init.set_zero(); w_init.set_zero();

	//compute the analytical solution of u velocity before rotation
	for (int k = 0; k < nk; k++) {
		for (int j = 0; j < nj; j++) {
			for (int i = 0; i < ni + 1; i++) {
				if (u_identifer_init(i, j, k) == CellType::Inside)
				{
					double y_coordinate = lower_vertex[1] + (j + 0.5) * dx_mm;
					double y_co = (y_coordinate - 0.6) / 2.47;
					double c = 1.2*(y_co*(1.0 - y_co));

					u_init(i, j, k) = c;
				}
			}
		}
	}
	Array3BT u_id = u_identifer_init;
	extrapolate(u_init, u_id);

	//compute the error of u velocity
	double u_error_avg = 0,u_error_total=0,u_error_deviation=0;
	int u_number = 0;
	double u_max_error = FLT_MIN, u_min_error = FLT_MAX;
	for (int k = 0; k < nk; k++) {
		for (int j = 0; j < nj; j++) {
			for (int i = 0; i < ni+1; i++) {
				if (u_identifer(i, j, k) == CellType::Inside)
				{
					//Only statistic the back part of the circle
					double x_co = lower_vertex[0] + i * dx_mm;
					double y_co = lower_vertex[1] + (j + 0.5) * dx_mm;
					double val1 = x_co * cos(al) - y_co * sin(al) - 3.45;
					double val2 = x_co * cos(al) - y_co * sin(al) - 6.05;

					if (val1 > 0&&val2<0)
					{
						Vec3d orientation(cos(al), sin(al), 0);
						Vec3d pnew = lower_vertex + Vec3d(i * dx_mm, (j + 0.5) * dx_mm, (k + 0.5) * dx_mm);
						Vec3d pold(pnew[0] * cos(al) - pnew[1] * sin(al), pnew[0] * sin(al) + pnew[1] * cos(al), pnew[2]);
						pold = pold - lower_vertex;

						Vec3d vel_p = get_velocity(u_init, v_init, w_init, pold);
						double vel_o = dot(vel_p, orientation);
						u_error(i, j, k) = (vel_o - u(i, j, k)) / 0.2;

						u_max_error = max(u_max_error, u_error(i, j, k));
						u_min_error = min(u_min_error, u_error(i, j, k));
						u_error_total += u_error(i, j, k);
						u_number++;
						std::cout << "this grid is: " << i << " " << j << " " << k << " " << vel_o << " " << u(i, j, k) << " " << u_error(i, j, k) << endl;
					}
				}
			}
		}
	}
	u_error_avg = u_error_total / u_number;
	for (int k = 0; k < nk; k++) {
		for (int j = 0; j < nj; j++) {
			for (int i = 0; i < ni + 1; i++) {
				if (u_identifer(i, j, k) == CellType::Inside)
				{
					//Only statistic the back part of the circle
					double x_co = lower_vertex[0] + i * dx_mm;
					double y_co = lower_vertex[1] + (j + 0.5) * dx_mm;
					double val1 = x_co * cos(al) - y_co * sin(al) - 3.45;
					double val2 = x_co * cos(al) - y_co * sin(al) - 6.05;

					if (val1 > 0 && val2 < 0)
					{
						u_error_deviation += sqr(u_error(i, j, k) - u_error_avg);
					}
				}
			}
		}
	}
	std::cout << "the max and min error of u velocity is: " << u_max_error << "  " << u_min_error << endl;
	std::cout << "the average error of u velocity is: " << u_error_avg << endl;
	std::cout << "the deviation of error of u velocity is: " << u_error_deviation/u_number << endl;
	std::cout << std::endl << std::endl;

	//compute the error of v velocity
	double v_error_avg = 0, v_error_total = 0, v_error_deviation = 0;
	int v_number = 0;
	double v_max_error = FLT_MIN, v_min_error = FLT_MAX;
	for (int k = 0; k < nk; k++) {
		for (int j = 0; j < nj + 1; j++) {
			for (int i = 0; i < ni; i++) {
				if (v_identifer(i, j, k) == CellType::Inside)
				{
					//Only statistic the back part of the circle
					double x_co = lower_vertex[0] + i * dx_mm;
					double y_co = lower_vertex[1] + (j + 0.5) * dx_mm;
					double val1 = x_co * cos(al) - y_co * sin(al) - 3.45;
					double val2 = x_co * cos(al) - y_co * sin(al) - 6.05;

					if (val1 > 0 && val2 < 0)
					{
						Vec3d orientation(sin(al), -cos(al), 0);
						Vec3d pnew = lower_vertex + Vec3d((i + 0.5) * dx_mm, j* dx_mm, (k + 0.5) * dx_mm);
						Vec3d pold(pnew[0] * cos(al) - pnew[1] * sin(al), pnew[0] * sin(al) + pnew[1] * cos(al), pnew[2]);
						pold = pold - lower_vertex;

						Vec3d vel_p = get_velocity(u_init, v_init, w_init, pold);
						double vel_o = -dot(vel_p, orientation);
						v_error(i, j, k) = (vel_o - v(i, j, k)) / 0.2;

						v_max_error = max(v_max_error, v_error(i, j, k));
						v_min_error = min(v_min_error, v_error(i, j, k));
						v_error_total += v_error(i, j, k);
						v_number++;
						std::cout << "this grid is: " << i << " " << j << " " << k << " " " " << vel_o << " " << v(i, j, k) << " " << v_error(i, j, k) << endl;
					}
				}
			}
		}
	}
	v_error_avg = v_error_total / v_number;
	for (int k = 0; k < nk; k++) {
		for (int j = 0; j < nj + 1; j++) {
			for (int i = 0; i < ni; i++) {
				if (v_identifer(i, j, k) == CellType::Inside)
				{
					//Only statistic the back part of the circle
					double x_co = lower_vertex[0] + i * dx_mm;
					double y_co = lower_vertex[1] + (j + 0.5) * dx_mm;
					double val1 = x_co * cos(al) - y_co * sin(al) - 3.45;
					double val2 = x_co * cos(al) - y_co * sin(al) - 6.05;

					if (val1 > 0 && val2 < 0)
					{
						v_error_deviation += sqr(v_error(i, j, k) - v_error_avg);
					}
				}
			}
		}
	}
	std::cout << "the max and min error of v velocity is: " << v_max_error << "  " << v_min_error << endl;
	std::cout << "the average error of v velocity is: " << v_error_avg << endl;
	std::cout << "the deviation of error of v velocity is: " << v_error_deviation / v_number << endl;
}

//Basic first order semi-Lagrangian advection of velocities
void Simple::advect(Array3d &velocity_u, Array3d &velocity_v, Array3d &velocity_w, double t)
{
	temp_u.resize(ni + 1, nj, nk);
	temp_v.resize(ni, nj + 1, nk);
	temp_w.resize(ni, nj, nk + 1);
	temp_u.assign(0);
	temp_v.assign(0);
	temp_w.assign(0);

	//semi-Lagrangian advection on u-component of velocity
	for (int k = 0; k < nk; ++k) for (int j = 0; j < nj; ++j) for (int i = 0; i < ni + 1; ++i)
	{
		Vec3d pos(i * dx_mm, (j + 0.5) * dx_mm, (k + 0.5) * dx_mm);
		pos = trace_rk2(velocity_u, velocity_v, velocity_w, pos, -t);
		temp_u(i, j, k) = get_velocity(velocity_u, velocity_v, velocity_w, pos)[0];
	}

	//semi-Lagrangian advection on v-component of velocity
	for (int k = 0; k < nk; ++k) for (int j = 0; j < nj + 1; ++j) for (int i = 0; i < ni; ++i)
	{
		Vec3d pos((i + 0.5) * dx_mm, j * dx_mm, (k + 0.5) * dx_mm);
		pos = trace_rk2(velocity_u, velocity_v, velocity_w, pos, -t);
		temp_v(i, j, k) = get_velocity(velocity_u, velocity_v, velocity_w, pos)[1];
	}

	//semi-Lagrangian advection on w-component of velocity
	for (int k = 0; k < nk + 1; ++k) for (int j = 0; j < nj; ++j) for (int i = 0; i < ni; ++i)
	{
		Vec3d pos((i + 0.5) * dx_mm, (j + 0.5) * dx_mm, k * dx_mm);
		pos = trace_rk2(velocity_u, velocity_v, velocity_w, pos, -t);
		temp_w(i, j, k) = get_velocity(velocity_u, velocity_v, velocity_w, pos)[2];
	}

	//move update velocities into u/v vectors
	velocity_u = temp_u;
	velocity_v = temp_v;
	velocity_w = temp_w;
}
//Apply RK2 to advect a point in the domain.
Vec3d Simple::trace_rk2(const Array3d& vol_u, const Array3d& vol_v, const Array3d& vol_w, const Vec3d& position, double t)
{
	Vec3d input = position;
	Vec3d velocity = get_velocity(vol_u, vol_v, vol_w, input);
	velocity = get_velocity(vol_u, vol_v, vol_w, input + 0.5 * t * velocity);
	input += t * velocity;
	return input;
}
//Interpolate velocity from the MAC grid.
Vec3d Simple::get_velocity(const Array3d& v_u, const Array3d& v_v, const Array3d& v_w, const Vec3d& position)
{
	//Interpolate the velocity from the u and v grids
	double u_value = interpolate_value(position / dx_mm - Vec3d(0, 0.5, 0.5), v_u);
	double v_value = interpolate_value(position / dx_mm - Vec3d(0.5, 0, 0.5), v_v);
	double w_value = interpolate_value(position / dx_mm - Vec3d(0.5, 0.5, 0), v_w);

	return Vec3d(u_value, v_value, w_value);
}


