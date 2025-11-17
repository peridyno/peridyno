#pragma once
#include "Array/ArrayList.h"

#include "STL/Pair.h"

#include "Matrix/Transform3x3.h"

#include "Collision/CollisionData.h"

#include "Topology/DiscreteElements.h"

#include "Algorithm/Reduction.h"

#include "Collision/Attribute.h"

#include <set>

#include <algorithm>

#include <random>

#include <optional>


namespace dyno 
{
	void ApplyTransform(
		DArrayList<Transform3f>& instanceTransform,
		const DArray<Vec3f>& translate,
		const DArray<Mat3f>& rotation,
		const DArray<Pair<uint, uint>>& binding,
		const DArray<int>& bindingtag);

	void updateVelocity(
		DArray<Attribute> attribute,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray<Vec3f> impulse,
		float linearDamping,
		float angularDamping,
		float dt
	);

	void updateGesture(
		DArray<Attribute> attribute,
		DArray<Vec3f> pos,
		DArray<Quat1f> rotQuat,
		DArray<Mat3f> rotMat,
		DArray<Mat3f> inertia,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray<Mat3f> inertia_init,
		float dt
	);

	void updateGestureNoSelf(
		DArray<Attribute> attribute,
		DArray<Vec3f> initPos,
		DArray<Vec3f> pos,
		DArray<Quat1f> initRotQuat,
		DArray<Quat1f> rotQuat,
		DArray<Mat3f> initRotMat,
		DArray<Mat3f> rotMat,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		float dt
	);

	void updateInitialGuess(
		DArray<Attribute> attribute,
		DArray<Vec3f> pos,
		DArray<Quat1f> rotQuat,
		DArray<Vec3f> normalPos,
		DArray<Quat1f> normalRotQuat,
		DArray<Mat3f> rotMat,
		DArray<Mat3f> inertia,
		DArray<Vec3f> velocity,
		DArray<Vec3f> pre_velocity,
		DArray<Vec3f> angular_velocity,
		DArray<Mat3f> inertia_init,
		bool hasGravity,
		Real Gravity,
		float dt
	);

	void updatePositionAndRotation(
		DArray<Vec3f> pos,
		DArray<Quat1f> rotQuat,
		DArray<Mat3f> rotMat,
		DArray<Mat3f> inertia,
		DArray<Mat3f> inertia_init,
		DArray<Vec3f> impulse_constrain
	);

	void calculateContactPoints(
		DArray<TContactPair<float>> contacts,
		DArray<int> contactCnt
	);


	void calculateJacobianMatrix(
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<Vec3f> pos,
		DArray<Mat3f> inertia,
		DArray<float> mass,
		DArray<Mat3f> rotMat,
		DArray<TConstraintPair<float>> constraints
	);

	void calculateJacobianMatrixForNJS(
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<Vec3f> pos,
		DArray<Mat3f> inertia,
		DArray<float> mass,
		DArray<Mat3f> rotMat,
		DArray<TConstraintPair<float>> constraints
	);


	void calculateEtaVectorForPJS(
		DArray<float> eta,
		DArray<Vec3f> J,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray<TConstraintPair<float>> constraints
	);

	void calculateEtaVectorForPJSBaumgarte(
		DArray<float> eta,
		DArray<Vec3f> J,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray<Vec3f> pos,
		DArray<Quat1f> rotation_q,
		DArray<TConstraintPair<float>> constraints,
		DArray<float> errors,
		float slop,
		float beta,
		uint substepping,
		float dt
	);

	void calculateEtaVectorWithERP(
		DArray<float> eta,
		DArray<Vec3f> J,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray<Vec3f> pos,
		DArray<Quat1f> rotation_q,
		DArray <TConstraintPair<float>> constraints,
		DArray<float> ERP,
		float slop,
		float dt
	);

	void calculateEtaVectorForPJSoft(
		DArray<float> eta,
		DArray<Vec3f> J,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray<Vec3f> pos,
		DArray<Quat1f> rotation_q,
		DArray <TConstraintPair<float>> constraints,
		float slop,
		float zeta,
		float hertz,
		float substepping,
		float dt
	);
	
	void calculateEtaVectorForNJS(
		DArray<float> eta,
		DArray<Vec3f> J,
		DArray<Vec3f> pos,
		DArray<Quat1f> rotation_q,
		DArray <TConstraintPair<float>> constraints,
		float slop,
		float beta
	);
	
	void setUpContactsInLocalFrame(
		DArray<TContactPair<float>> contactsInLocalFrame,
		DArray<TContactPair<float>> contactsInGlobalFrame,
		DArray<Vec3f> pos,
		DArray<Mat3f> rotMat
	);
	
	void setUpContactAndFrictionConstraints(
		DArray<TConstraintPair<float>> constraints,
		DArray<TContactPair<float>> contactsInLocalFrame,
		DArray<Vec3f> pos,
		DArray<Mat3f> rotMat,
		bool hasFriction
	);

	void setUpContactAndFrictionConstraintsBlock(
		DArray<TConstraintPair<float>> constraints,
		DArray<TContactPair<float>> contactsInLocalFrame,
		DArray<Vec3f> pos,
		DArray<Mat3f> rotMat
	);

	void setUpContactAndFrictionConstraintForces(
		DArray<TConstraintForce<float>> constraintForces,
		DArray<TContactPair<float>> contactsInLocalFrame,
		DArray<Vec3f> pos,
		DArray<Mat3f> rotMat,
		bool hasFriction
	);
	
	void setUpContactConstraints(
		DArray<TConstraintPair<float>> constraints,
		DArray<TContactPair<float>> contactsInLocalFrame,
		DArray<Vec3f> pos,
		DArray<Mat3f> rotMat
	);

	void setUpBallAndSocketJointConstraints(
		DArray<TConstraintPair<float>> constraints,
		DArray<BallAndSocketJoint<float>> joints,
		DArray<Vec3f> pos,
		DArray<Mat3f> rotMat,
		int begin_index
	);
	
	void setUpSliderJointConstraints(
		DArray<TConstraintPair<float>> constraints,
		DArray<SliderJoint<float>> joints,
		DArray<Vec3f> pos,
		DArray<Mat3f> rotMat,
		DArray<Quat1f> rotQuat,
		int begin_index
	);

	void setUpHingeJointConstraints(
		DArray<TConstraintPair<float>> constraints,
		DArray<HingeJoint<float>> joints,
		DArray<Vec3f> pos,
		DArray<Mat3f> rotMat,
		DArray<Quat1f> rotation_q,
		int begin_index
	);

	void setUpFixedJointConstraints(
		DArray<TConstraintPair<float>>& constraints,
		DArray<FixedJoint<float>>& joints,
		DArray<Mat3f>& rotMat,
		DArray<Quat1f>& rotQuat,
		int begin_index
	);

	void setUpPointJointConstraints(
		DArray<TConstraintPair<float>> constraints,
		DArray<PointJoint<float>> joints,
		DArray<Vec3f> pos,
		int begin_index
	);

	void calculateK(
		DArray<TConstraintPair<float>> constraints,
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<Vec3f> pos,
		DArray<Mat3f> inertia,
		DArray<float> mass,
		DArray<float> K_1,
		DArray<Mat2f> K_2,
		DArray<Mat3f> K_3
	);

	void calculateKBlock(
		DArray<TConstraintPair<float>> constraints,
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<Vec3f> pos,
		DArray<Mat3f> inertia,
		DArray<float> mass,
		DArray<float> K_1,
		DArray<Mat2f> K_2,
		DArray<Mat3f> K_3
	);

	void calculateKWithCFM(
		DArray<TConstraintPair<float>> constraints,
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<Vec3f> pos,
		DArray<Mat3f> inertia,
		DArray<float> mass,
		DArray<float> K_1,
		DArray<Mat2f> K_2,
		DArray<Mat3f> K_3,
		DArray<float> CFM
	);


	void JacobiIteration(
		DArray<float> lambda,
		DArray<Vec3f> impulse,
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<float> eta,
		DArray<TConstraintPair<float>> constraints,
		DArray<int> nbq,
		DArray<float> K_1,
		DArray<Mat2f> K_2,
		DArray<Mat3f> K_3,
		DArray<float> mass,
		DArray<float> fricCoeffs,
		float mu,
		float g,
		float dt
	);

	void JacobiIterationForCFM(
		DArray<float> lambda,
		DArray<Vec3f> impulse,
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<float> eta,
		DArray<TConstraintPair<float>> constraints,
		DArray<int> nbq,
		DArray<float> K_1,
		DArray<Mat2f> K_2,
		DArray<Mat3f> K_3,
		DArray<float> mass,
		DArray<float> CFM,
		float mu,
		float g,
		float dt
	);

	void JacobiIterationStrict(
		DArray<float> lambda,
		DArray<Vec3f> impulse,
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<float> eta,
		DArray<TConstraintPair<float>> constraints,
		DArray<int> nbq,
		DArray<float> d,
		DArray<float> mass,
		float mu,
		float g,
		float dt
	);

	void JacobiIterationForSoft(
		DArray<float> lambda,
		DArray<Vec3f> impulse,
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<float> eta,
		DArray<TConstraintPair<float>> constraints,
		DArray<int> nbq,
		DArray<float> K_1,
		DArray<Mat2f> K_2,
		DArray<Mat3f> K_3,
		DArray<float> mass,
		DArray<float> mu,
		float g,
		float dt,
		float zeta,
		float hertz
	);

	void JacobiIterationForSoftBlock(
		DArray<float> lambda,
		DArray<Vec3f> impulse,
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<float> eta,
		DArray<TConstraintPair<float>> constraints,
		DArray<int> nbq,
		DArray<float> K_1,
		DArray<Mat2f> K_2,
		DArray<Mat3f> K_3,
		DArray<float> mass,
		DArray<float> mu,
		float g,
		float dt,
		float zeta,
		float hertz
	);

	void JacobiIterationForNJS(
		DArray<float> lambda,
		DArray<Vec3f> impulse,
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<float> eta,
		DArray<TConstraintPair<float>> constraints,
		DArray<int> nbq,
		DArray<float> K_1,
		DArray<Mat2f> K_2,
		DArray<Mat3f> K_3
	);

	void setUpGravity(
		DArray<Vec3f> impulse_ext,
		float g,
		float dt
	);


	Real checkOutError(
		DArray<Vec3f> J,
		DArray<Vec3f> mImpulse,
		DArray<TConstraintPair<float>> constraints,
		DArray<float> eta
	);

	void calculateDiagnals(
		DArray<float> d,
		DArray<Vec3f> J,
		DArray<Vec3f> B
	);

	void preConditionJ(
		DArray<Vec3f> J,
		DArray<float> d,
		DArray<float> eta
	);

	bool saveVectorToFile(
		const std::vector<float>& vec,
		const std::string& filename
	);


	void calculateEtaVectorForRelaxation(
		DArray<float> eta,
		DArray<Vec3f> J,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray <TConstraintPair<float>> constraints
	);

	double checkOutErrors(
		DArray<float> errors
	);
	


	void calculateMatrixA(
		DArray<Vec3f> &J,
		DArray<Vec3f> &B,
		DArray<float> &A,
		DArray<TConstraintPair<float>> &constraints,
		float k
	);

	bool saveMatrixToFile(
		DArray<float> &Matrix,
		int n,
		const std::string& filename
	);

	bool saveVectorToFile(
		DArray<float>& vec,
		const std::string& filename
	);

	void vectorSub(
		DArray<float> &ans,
		DArray<float> &subtranhend,
		DArray<float> &minuend,
		DArray<TConstraintPair<float>> &constraints
	);

	void vectorAdd(
		DArray<float>& ans,
		DArray<float>& v1,
		DArray<float>& v2,
		DArray<TConstraintPair<float>>& constraints
	);

	void vectorMultiplyScale(
		DArray<float> &ans,
		DArray<float> &initialVec,
		float scale,
		DArray<TConstraintPair<float>>& constraints
	);

	void vectorClampSupport(
		DArray<float> v,
		DArray<TConstraintPair<float>> constraints
	);

	void vectorClampFriction(
		DArray<float> v,
		DArray<TConstraintPair<float>> constraints,
		int contact_size,
		float mu
	);

	void matrixMultiplyVec(
		DArray<Vec3f> &J,
		DArray<Vec3f> &B,
		DArray<float> &lambda,
		DArray<float> &ans,
		DArray<TConstraintPair<float>> &constraints,
		int bodyNum
	);

	float vectorNorm(
		DArray<float> &a,
		DArray<float> &b
	);

	void vectorMultiplyVector(
		DArray<float>& v1,
		DArray<float>& v2,
		DArray<float>& ans,
		DArray<TConstraintPair<float>>& constraints
	);

	void calculateImpulseByLambda(
		DArray<float> lambda,
		DArray<TConstraintPair<float>> constraints,
		DArray<Vec3f> impulse,
		DArray<Vec3f> B
	);

	void preconditionedResidual(
		DArray<float> &residual,
		DArray<float> &ans,
		DArray<float> &k_1,
		DArray<Mat2f> &k_2,
		DArray<Mat3f> &k_3,
		DArray<TConstraintPair<float>> &constraints
	);

	void buildCFMAndERP(
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<TConstraintPair<float>> constraints,
		DArray<float> CFM,
		DArray<float> ERP,
		float hertz,
		float zeta,
		float dt
	);

	void calculateLinearSystemLHS(
		DArray<Vec3f>& J,
		DArray<Vec3f>& B,
		DArray<Vec3f>& impulse,
		DArray<float>& lambda,
		DArray<float>& ans,
		DArray<float>& CFM,
		DArray<TConstraintPair<float>>& constraints
	);


	class DynamicGraphColoring {
	public:
		/**
		 * @brief Default constructor.
		 */
		DynamicGraphColoring();

		/**
		 * @brief Initializes the graph with a given number of vertices and a list of edges.
		 * @param num_v The number of vertices in the graph.
		 * @param initial_edges A vector of pairs representing the initial edges.
		 */
		void initializeGraph(int num_v, const std::set<std::pair<int, int>>& initial_edges);

		/**
		 * @brief Performs the initial coloring of the graph.
		 */
		void performInitialColoring();

		/**
		 * @brief Applies a batch of edge additions and deletions to the graph and resolves conflicts.
		 * @param add_edges A vector of pairs representing edges to be added.
		 * @param delete_edges A vector of pairs representing edges to be deleted.
		 */
		void applyBatchUpdate(const std::set<std::pair<int, int>>& add_edges, const std::set<std::pair<int, int>>& delete_edges);

		/**
		 * @brief Gets the current coloring of the vertices.
		 * @return A const reference to the vector of colors.
		 */
		const std::vector<int>& getColors() const { return colors; }

		const std::vector<std::vector<int>>& getCategories() const { return categories; }

		/**
		 * @brief Gets the number of colors currently used.
		 * @return The number of colors.
		 */
		int getNumColors() const { return num_colors; }

		/**
		 * @brief Checks if the graph structure has been initialized.
		 * @return True if initializeGraph() has been called, false otherwise.
		 */
		bool isGraphInitialized() const {
			return graph_initialized_;
		}

	private:
		std::vector<std::vector<int>> graph;      // Adjacency list representation of the graph
		std::vector<int> colors;                  // Stores the color of each vertex
		std::vector<std::vector<int>> categories; // Vertices grouped by color
		int num_vertices;                         // Number of vertices
		int num_colors;                           // Number of colors used
		bool graph_initialized_ = false;

		/**
		 * @brief Rebuilds the 'categories' data structure based on the current 'colors'.
		 */
		void rebuildCategories();

		/**
		 * @brief Finds a new valid color for a given node.
		 * @param node The vertex to find a new color for.
		 * @return An optional integer representing the new color. Returns an empty optional if no new color is found.
		 */
		std::optional<int> findNewColorFor(int node);

		/**
		 * @brief Checks if a node is in conflict (shares a color with a neighbor).
		 * @param node The vertex to check.
		 * @return True if the node is in conflict, false otherwise.
		 */
		bool isConflictNode(int node);

		/**
		 * @brief Updates the color of a node and the corresponding data structures.
		 * @param node The vertex to update.
		 * @param old_color The previous color of the vertex.
		 * @param new_color The new color for the vertex.
		 */
		void updateNodeColor(int node, int old_color, int new_color);

		// --- Helper functions for coloring ---

		std::vector<int> orderedGreedyColoring(const std::vector<std::vector<int>>& graph);

		void balanceColoring(const std::vector<std::vector<int>>& graph, std::vector<int>& colors, float goal_ratio = 1.5, int maxAttempts = 1000);;
	};

	void constraintsMappingToEdges(DArray<TConstraintPair<float>> constraints, std::vector<std::pair<int, int>>& edges);

	void constraintForceMappingToEdges(
		DArray<TConstraintForce<float>> constraintForces,
		std::set<std::pair<int, int>>& edges
	);

	void reduceContacts(DArray<TContactPair<float>>& contacts, float normalThreshold, float penetrationThreshold);

	void reduceContacts_Optimized(DArray<TContactPair<float>>& contacts, float normalThreshold, float penetrationThreshold);
}
