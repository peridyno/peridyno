#ifndef SIMPLE_H
#define SIMPLE_H

#include "array3.h"
#include "vec.h"

#include<Eigen/Sparse>
#include <vector>
#include <stdio.h>
#include <string>

#include "DistanceField3D.h"

enum CellType
{
	Undefined = 0,
	Inside,
	Inlet1,
	Inlet2,
	Outlet1,
	Outlet2,
	Static
};


class Simple
{
	typedef Array3<CellType, Array1<CellType> > Array3BT;

public:
	//double pressure_indicator;
	void initialize(double dx_, std::string fluid, std::string inlet1, std::string inlet2, std::string outlet1, std::string outlet2, double alpha, double theta);
	void initialize(Vec3d pos, double dx_, int ni_, int nj_, int nk_, mfd::DistanceField3D& fluidSDF, mfd::DistanceField3D& inletSDF, mfd::DistanceField3D& outletSDF,double alpha);
	void initialize(double dx_, std::string in_model, std::string in_centerline);
	void advance(std::string path, int frame, double dt, double alpha,double theta);
	void export_status(std::string path, int frame);
	void statistics(std::string face1,std::string face2,double alpha, double theta);
	double statistics_pressure(std::string face);
	Vec3d statistics_vel(std::string face);
	bool import_status(std::string path);
	//export velocity field and pressure velocity in .vtk file
	void export_velocity_vtk(std::string path);
	void export_pressure_vtk(std::string path);
	//compute and cout date for pressure map
	void face_vertex_pressure(std::string path1, std::string path2, double zero1, double zero2, std::string model_sim, std::string pout_path);

private:
	//
	//Grid dimensions
	int ni, nj, nk;
	double dx, dx_mm;
	Vec3d lower_vertex;
	//Static geometry representation(if models not change, these variables will not change)
	Array3d liquid_phi,u_liquid_phi,v_liquid_phi, w_liquid_phi;
	Array3BT p_identifer,u_identifer,v_identifer,w_identifer;

	//Fluid velocity and pressure and coefficient
	Array3d u, v, w;
	Array3d u_starstar, v_starstar, w_starstar;
	Array3d u_prime, v_prime, w_prime;
	Array3d p_star, p_prime,p;
	Array3d temp_u, temp_v, temp_w;//used during advection

	double convergence_indicator;

	//inlet condition
	Vec3d inlet_direction1, inlet_direction2;
	double inlet_magnitude1, inlet_magnitude2;
	//parameter
	double total_time = 0.0;
	double blood_density = 1060.0;
	double kinetic_coefficient = 0.004;
	double convergence_standard = 0.0005;
	int number_iterations =5;

	//centerline
	vector<Vec3d> centerline_points;
	int inlet_intersection, outlet_intersection;
	vector<int> pv, lpv, rpv, sv, smv;
	double inlet_standard = 10;//standard distance from inlet intersection, unit:mm
	double outlet_standard1 = 12;//standard distance from outlet intersection of LPV, unit:mm
	double outlet_standard2 = 8;//standard distance from outlet intersection of RPV, unit:mm
	Vec3d outlet_direction1, outlet_direction2;
	//Solver data
	typedef Eigen::SparseMatrix<double> SpMat;
	typedef Eigen::Triplet<double> TripletD;
	Eigen::VectorXd b;
	SpMat A;

	//functions
	void initialize_sdf(mfd::DistanceField3D& fSDF, mfd::DistanceField3D& inSDF1, mfd::DistanceField3D& inSDF2, mfd::DistanceField3D& outSDF1, mfd::DistanceField3D& outSDF2, Array3d& p_liq, Array3d& u_liq, Array3d& v_liq, Array3d& w_liq, Array3BT& p_id, Array3BT& u_id, Array3BT& v_id, Array3BT& w_id);
	void initialize_sdf(mfd::DistanceField3D& fSDF);
	void classifyBoundary(CellType& ct, Vec3d pos, mfd::DistanceField3D& inletSDF1, mfd::DistanceField3D& inletSDF2, mfd::DistanceField3D& outletSDF1, mfd::DistanceField3D& outletSDF2);
	double u_boundary(Vec3d& in_dir, double& in_mag, double al,double the, int grid_j, int grid_k);
	double v_boundary(Vec3d& in_dir, double& in_mag, double al,double the);
	double w_boundary(Vec3d& in_dir, double& in_mag, double al,double the);
	double p_boundary();
	void solve_velocity_component(Array3d& vel, Array3d& vel_ss, Array3BT& identifier, Array3d& liquid_phi, double dt, double alpha,double theta);
	void solve_pressure(Array3d& pressure, Array3d& u_in, Array3d& v_in, Array3d& w_in, double dt, double alpha,double theta);
	void update_velocity(Array3d& u, Array3d& v, Array3d& w, Array3d& pressure, double dt, double alpha,double theta);
	double cfl();
	void compute_velocity(double dt);
	void extrapolate(Array3d& grid, Array3BT iden);
	void statistics_velocity(Array3BT& identifer, Array3d& velocity, Vec3d& orientation, double alpha, double theta);
	void statistics(double alpha);
	void advect(Array3d &volecity_u, Array3d &volecity_v, Array3d &volecity_w, double t);
	Vec3d trace_rk2(const Array3d& vol_u, const Array3d& vol_v, const Array3d& vol_w, const Vec3d& position, double t);
	Vec3d get_velocity(const Array3d& v_u, const Array3d& v_v, const Array3d& v_w, const Vec3d& position);
	bool centerline(std::string in_centerline);

	//Static geometry representation before rotation
	Array3d liquid_phi_init, u_liquid_phi_init, v_liquid_phi_init, w_liquid_phi_init;
	Array3BT p_identifer_init, u_identifer_init, v_identifer_init, w_identifer_init;
	Array3d u_init, v_init, w_init;
};

#endif
