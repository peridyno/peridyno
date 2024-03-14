#include "EdgeInteraction.h"
#include <iostream>
#include <OrbitCamera.h>

namespace dyno
{
    template <typename TDataType>
    EdgeInteraction<TDataType>::EdgeInteraction() {
        this->ray1 = TRay3D<Real>();
        this->ray2 = TRay3D<Real>();
        this->isPressed = false;
        this->outOtherEdgeSet()->setDataPtr(std::make_shared<EdgeSet3f>());
        this->outOtherEdgeSet()->getDataPtr()->getEdges().resize(0);
        this->outSelectedEdgeSet()->setDataPtr(std::make_shared<EdgeSet3f>());
        this->outSelectedEdgeSet()->getDataPtr()->getEdges().resize(0);

        this->addKernel("EdgeRay",
                        std::make_shared<VkProgram>(BUFFER(Coord), BUFFER(int), BUFFER(int), BUFFER(Real), BUFFER(Edge),
                                                    CONSTANT(uint), CONSTANT(TRay3D<Real>), CONSTANT(Real)));
        this->kernel("EdgeRay")->load(VkSystem::instance()->getAssetPath() /
                                      "shaders/glsl/interaction/EdgeRay.comp.spv");

        this->addKernel("Nearest",
                        std::make_shared<VkProgram>(BUFFER(int), BUFFER(int), CONSTANT(uint), CONSTANT(int)));
        this->kernel("Nearest")->load(VkSystem::instance()->getAssetPath() /
                                      "shaders/glsl/interaction/Nearest.comp.spv");

        this->addKernel("EdgeAssignOut",
                        std::make_shared<VkProgram>(BUFFER(Edge), BUFFER(Edge), BUFFER(Edge), BUFFER(int), BUFFER(int),
                                                    BUFFER(int), BUFFER(int), CONSTANT(uint)));
        this->kernel("EdgeAssignOut")
            ->load(getSpvFile("shaders/glsl/interaction/EdgeAssignOut.comp.spv"));

        this->addKernel("EdgeBox", std::make_shared<VkProgram>(
                                       BUFFER(Coord), BUFFER(Edge), BUFFER(int), BUFFER(int), CONSTANT(uint),
                                       CONSTANT(TPlane3D<Real>), CONSTANT(TPlane3D<Real>), CONSTANT(TPlane3D<Real>),
                                       CONSTANT(TPlane3D<Real>), CONSTANT(TRay3D<Real>), CONSTANT(Real)));
        this->kernel("EdgeBox")->load(VkSystem::instance()->getAssetPath() /
                                      "shaders/glsl/interaction/EdgeBox.comp.spv");
        this->addKernel("Merge", std::make_shared<VkProgram>(BUFFER(int), BUFFER(int), BUFFER(int), BUFFER(int),
            CONSTANT(uint), CONSTANT(int)));
        this->kernel("Merge")
            ->load(getSpvFile("shaders/glsl/interaction/Merge.comp.spv"));
    }

    template <typename TDataType>
    void EdgeInteraction<TDataType>::onEvent(PMouseEvent event) {
        if (!event.altKeyPressed()) {
            if (camera == nullptr) {
                this->camera = event.camera;
            }
            this->varToggleMultiSelect()->setValue(false);
            if (event.shiftKeyPressed() || event.controlKeyPressed()) {
                this->varToggleMultiSelect()->setValue(true);
                if (event.shiftKeyPressed() && !event.controlKeyPressed()) {
                    this->varMultiSelectionType()->getDataPtr()->setCurrentKey(0);
                }
                else if (!event.shiftKeyPressed() && event.controlKeyPressed()) {
                    this->varMultiSelectionType()->getDataPtr()->setCurrentKey(1);
                    ;
                }
                else if (event.shiftKeyPressed() && event.controlKeyPressed()) {
                    this->varMultiSelectionType()->getDataPtr()->setCurrentKey(2);
                    ;
                }
            }
            if (event.actionType == AT_PRESS) {
                this->camera = event.camera;
                this->isPressed = true;
                // printf("Mouse pressed: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x,
                // event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y,
                // event.ray.direction.z);
                this->ray1.origin = event.ray.origin;
                this->ray1.direction = event.ray.direction;
                this->x1 = event.x;
                this->y1 = event.y;
                if (this->varEdgePickingType()->getValue() == PickingTypeSelection::Both ||
                    this->varEdgePickingType()->getValue() == PickingTypeSelection::Click)
                {
                    this->calcIntersectClick();
                    this->printInfoClick();
                }
            }
            else if (event.actionType == AT_RELEASE) {
                this->isPressed = false;
                // printf("Mouse released: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x,
                // event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y,
                // event.ray.direction.z);
                this->ray2.origin = event.ray.origin;
                this->ray2.direction = event.ray.direction;
                this->x2 = event.x;
                this->y2 = event.y;
                if (this->varToggleMultiSelect()->getValue() && this->varTogglePicker()->getValue()) {
                    this->mergeIndex();
                    this->printInfoDragRelease();
                }
            }
            else {
                // printf("Mouse repeated: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x,
                // event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y,
                // event.ray.direction.z);
                if (this->isPressed) {
                    this->ray2.origin = event.ray.origin;
                    this->ray2.direction = event.ray.direction;
                    this->x2 = event.x;
                    this->y2 = event.y;
                    if (this->x2 == this->x1 && this->y2 == this->y1) {
                        if (this->varEdgePickingType()->getValue() == PickingTypeSelection::Both ||
                            this->varEdgePickingType()->getValue() == PickingTypeSelection::Click)
                            this->calcIntersectClick();
                    }
                    else {
                        if (this->varEdgePickingType()->getValue() == PickingTypeSelection::Both ||
                            this->varEdgePickingType()->getValue() == PickingTypeSelection::Drag)
                        {
                            this->calcIntersectDrag();
                            this->printInfoDragging();
                        }
                    }
                }
            }
        }
    }

    template <typename TDataType>
    void EdgeInteraction<TDataType>::calcEdgeIntersectClick() {
        VkCompContext::Holder holder;
        holder.delaySubmit(true);

        EdgeSet<TDataType> initialEdgeSet = this->inInitialEdgeSet()->getData();
        DArray<Edge> edges = initialEdgeSet.getEdges();
        DArray<Coord> points = initialEdgeSet.getPoints();

        this->tempNumT = edges.size();
        DArray<int> intersected(edges.size());
        vkFill(*intersected.handle(), 0);
        DArray<int> unintersected(edges.size());

        DArray<Real> lineDistance(edges.size());

        VkConstant<uint> vk_num {edges.size()};
        VkConstant<TRay3D<Real>> vk_ray {this->ray1};
        VkConstant<Real> vk_radius {this->varInteractionRadius()->getData()};
        this->kernel("EdgeRay")->submit(vkDispatchSize(vk_num, 64), points.handle(), intersected.handle(),
                                       unintersected.handle(), lineDistance.handle(), edges.handle(), &vk_num, &vk_ray,
                                       &vk_radius);
        int min_index = mMin.reduce(*lineDistance.handle());

        vk_num.setValue(intersected.size());
        VkConstant<int> vk_min_index {min_index};
        this->kernel("Nearest")->submit(vkDispatchSize(vk_num, 64), intersected.handle(), unintersected.handle(),
                                       &vk_num, &vk_min_index);

        this->tempEdgeIntersectedIndex.assign(intersected);
        if (this->varToggleMultiSelect()->getData()) {
            if (this->edgeIntersectedIndex.size() == 0) {
                this->edgeIntersectedIndex.resize(edges.size());
            }
            DArray<int> outIntersected(intersected.size());
            DArray<int> outUnintersected(unintersected.size());

            VkConstant<int> vk_select_type {(int)this->varMultiSelectionType()->getValue().currentKey()};
            this->kernel("Merge")->submit(vkDispatchSize(vk_num, 64), this->edgeIntersectedIndex.handle(),
                                         intersected.handle(), outIntersected.handle(), outUnintersected.handle(),
                                         &vk_num, &vk_select_type);

            intersected.assign(outIntersected);
            unintersected.assign(outUnintersected);
        }
        else {
            this->edgeIntersectedIndex.assign(intersected);
        }

        DArray<int> intersected_o;
        intersected_o.assign(intersected);

        int intersected_size = mReduce.reduce(*intersected.handle());
        DArray<int> outEdgeIndex;
        outEdgeIndex.resize(intersected_size);
        mScan.scan(*intersected.handle(), *intersected.handle(), VkScan<int>::Exclusive);
        DArray<Edge> intersected_edges(intersected_size);

        int unintersected_size = mReduce.reduce(*unintersected.handle());
        mScan.scan(*unintersected.handle(), *unintersected.handle(), VkScan<int>::Exclusive);
        DArray<Edge> unintersected_edges(unintersected_size);

        this->tempNumS = intersected_size;

        vk_num.setValue(edges.size());
        this->kernel("EdgeAssignOut")
            ->submit(vkDispatchSize(vk_num, 64), edges.handle(), intersected_edges.handle(),
                    unintersected_edges.handle(), outEdgeIndex.handle(), intersected.handle(), unintersected.handle(),
                    intersected_o.handle(), &vk_num);

        this->outSelectedEdgeSet()->getDataPtr()->copyFrom(initialEdgeSet);
        this->outSelectedEdgeSet()->getDataPtr()->setEdges(intersected_edges);
        this->outOtherEdgeSet()->getDataPtr()->copyFrom(initialEdgeSet);
        this->outOtherEdgeSet()->getDataPtr()->setEdges(unintersected_edges);
        if (this->varToggleIndexOutput()->getValue()) {
            this->outEdgeIndex()->getDataPtr()->assign(outEdgeIndex);
        }
        else {
            this->outEdgeIndex()->getDataPtr()->assign(intersected_o);
        }
    }

    template <typename TDataType>
    void EdgeInteraction<TDataType>::calcEdgeIntersectDrag() {
        VkCompContext::Holder holder;
        holder.delaySubmit(true);

        if (x1 == x2) {
            x2 += 1.0f;
        }
        if (y1 == y2) {
            y2 += 1.0f;
        }
        TRay3D<Real> ray1 = this->camera->castRayInWorldSpace((float)x1, (float)y1);
        TRay3D<Real> ray2 = this->camera->castRayInWorldSpace((float)x2, (float)y2);
        TRay3D<Real> ray3 = this->camera->castRayInWorldSpace((float)x1, (float)y2);
        TRay3D<Real> ray4 = this->camera->castRayInWorldSpace((float)x2, (float)y1);

        VkConstant<TPlane3D<Real>> plane13 {TPlane3D<Real>(ray1.origin, ray1.direction.cross(ray3.direction))};
        VkConstant<TPlane3D<Real>> plane42 {TPlane3D<Real>(ray2.origin, ray2.direction.cross(ray4.direction))};
        VkConstant<TPlane3D<Real>> plane14 {TPlane3D<Real>(ray4.origin, ray1.direction.cross(ray4.direction))};
        VkConstant<TPlane3D<Real>> plane32 {TPlane3D<Real>(ray3.origin, ray2.direction.cross(ray3.direction))};

        EdgeSet3f initialEdgeSet = this->inInitialEdgeSet()->getData();
        DArray<Edge> edges = initialEdgeSet.getEdges();
        DArray<Coord> points = initialEdgeSet.getPoints();
        DArray<int> intersected(edges.size());
        vkFill(*intersected.handle(), 0);
        DArray<int> unintersected(edges.size());
        this->tempNumT = edges.size();

        VkConstant<uint> vk_num {edges.size()};
        VkConstant<TRay3D<Real>> vk_ray {this->ray1};
        VkConstant<Real> vk_radius {this->varInteractionRadius()->getValue()};
        this->kernel("EdgeBox")
            ->submit(vkDispatchSize(vk_num, 64), points.handle(), edges.handle(), intersected.handle(), unintersected.handle(), &vk_num,
                    &plane13, &plane42, &plane14, &plane32, &vk_ray, &vk_radius);

        this->tempEdgeIntersectedIndex.assign(intersected);

        if (this->varToggleMultiSelect()->getData()) {
            if (this->edgeIntersectedIndex.size() == 0) {
                this->edgeIntersectedIndex.resize(edges.size());
            }
            DArray<int> outIntersected(intersected.size());
            DArray<int> outUnintersected(unintersected.size());

            VkConstant<int> vk_select_type {(int)this->varMultiSelectionType()->getValue().currentKey()};
            this->kernel("Merge")->submit(vkDispatchSize(vk_num, 64), this->edgeIntersectedIndex.handle(),
                                         intersected.handle(), outIntersected.handle(), outUnintersected.handle(),
                                         &vk_num, &vk_select_type);

            intersected.assign(outIntersected);
            unintersected.assign(outUnintersected);
        }
        else {
            this->edgeIntersectedIndex.assign(intersected);
        }

        DArray<int> intersected_o;
        intersected_o.assign(intersected);

        int intersected_size = mReduce.reduce(*intersected.handle());
        DArray<int> outEdgeIndex(intersected_size);
        mScan.scan(*intersected.handle(), *intersected.handle(), VkScan<int>::Exclusive);
        DArray<Edge> intersected_edges(intersected_size);

        int unintersected_size = mReduce.reduce(*unintersected.handle());
        mScan.scan(*unintersected.handle(), *unintersected.handle(), VkScan<int>::Exclusive);
        DArray<Edge> unintersected_edges(unintersected_size);

        vk_num.setValue(edges.size());
        this->kernel("EdgeAssignOut")
            ->submit(vkDispatchSize(vk_num, 64), edges.handle(), intersected_edges.handle(),
                    unintersected_edges.handle(), outEdgeIndex.handle(), intersected.handle(), unintersected.handle(),
                    intersected_o.handle(), &vk_num);

        this->tempNumS = intersected_size;
        this->outSelectedEdgeSet()->getDataPtr()->copyFrom(initialEdgeSet);
        this->outSelectedEdgeSet()->getDataPtr()->setEdges(intersected_edges);
        this->outOtherEdgeSet()->getDataPtr()->copyFrom(initialEdgeSet);
        this->outOtherEdgeSet()->getDataPtr()->setEdges(unintersected_edges);
        if (this->varToggleIndexOutput()->getValue()) {
            this->outEdgeIndex()->getDataPtr()->assign(outEdgeIndex);
        }
        else {
            this->outEdgeIndex()->getDataPtr()->assign(intersected_o);
        }
    }

    template <typename TDataType>
    void EdgeInteraction<TDataType>::mergeIndex() {
        VkCompContext::Holder holder;
        holder.delaySubmit(true);

		EdgeSet<TDataType> initialEdgeSet = this->inInitialEdgeSet()->getData();
		DArray<Edge> edges = initialEdgeSet.getEdges();
		DArray<Coord> points = initialEdgeSet.getPoints();
		DArray<int> intersected(edges.size());
		DArray<int> unintersected(edges.size());
		this->tempNumT = edges.size();

		DArray<int> outIntersected(intersected.size());
		DArray<int> outUnintersected(unintersected.size());

        VkConstant<uint> vk_num {this->edgeIntersectedIndex.size()};
        if (this->varToggleMultiSelect()->getData()) {
            VkConstant<int> vk_select_type{ (int)this->varMultiSelectionType()->getValue().currentKey() };
            this->kernel("Merge")->submit(vkDispatchSize(vk_num, 64), this->edgeIntersectedIndex.handle(),
                this->tempEdgeIntersectedIndex.handle(), outIntersected.handle(), outUnintersected.handle(),
                &vk_num, &vk_select_type);
        }
		intersected.assign(outIntersected);
		unintersected.assign(outUnintersected);
		this->edgeIntersectedIndex.assign(intersected);

        DArray<int> intersected_o;
        intersected_o.assign(intersected);

        int intersected_size = mReduce.reduce(*intersected.handle());
        DArray<int> outEdgeIndex(intersected_size);
        mScan.scan(*intersected.handle(), *intersected.handle(), VkScan<int>::Exclusive);
        DArray<Edge> intersected_edges(intersected_size);

        int unintersected_size = mReduce.reduce(*unintersected.handle());
        mScan.scan(*unintersected.handle(), *unintersected.handle(), VkScan<int>::Exclusive);
        DArray<Edge> unintersected_edges(unintersected_size);

        vk_num.setValue(edges.size());
        this->kernel("EdgeAssignOut")
            ->submit(vkDispatchSize(vk_num, 64), edges.handle(), intersected_edges.handle(),
                    unintersected_edges.handle(), outEdgeIndex.handle(), intersected.handle(), unintersected.handle(),
                    intersected_o.handle(), &vk_num);

		this->tempNumS = intersected_size;
		this->outSelectedEdgeSet()->getDataPtr()->copyFrom(initialEdgeSet);
		this->outSelectedEdgeSet()->getDataPtr()->setEdges(intersected_edges);
		this->outOtherEdgeSet()->getDataPtr()->copyFrom(initialEdgeSet);
		this->outOtherEdgeSet()->getDataPtr()->setEdges(unintersected_edges);
		if (this->varToggleIndexOutput()->getData())
		{
			this->outEdgeIndex()->getDataPtr()->assign(outEdgeIndex);
		}
		else
		{
			this->outEdgeIndex()->getDataPtr()->assign(intersected_o);
		}
    }

    template <typename TDataType>
    void EdgeInteraction<TDataType>::printInfoClick() {
        std::cout << "----------edge picking: click----------" << std::endl;
        std::cout << "multiple picking: " << this->varToggleMultiSelect()->getValue() << std::endl;
        std::cout << "Interation radius:" << this->varInteractionRadius()->getValue() << std::endl;
        std::cout << "selected num/ total num:" << this->tempNumS << "/" << this->tempNumT << std::endl;
    }

    template <typename TDataType>
    void EdgeInteraction<TDataType>::printInfoDragging() {
        std::cout << "----------edge picking: dragging----------" << std::endl;
        std::cout << "multiple picking: " << this->varToggleMultiSelect()->getValue() << std::endl;
        std::cout << "Interation radius:" << this->varInteractionRadius()->getValue() << std::endl;
        std::cout << "selected num/ total num:" << this->tempNumS << "/" << this->tempNumT << std::endl;
    }

    template <typename TDataType>
    void EdgeInteraction<TDataType>::printInfoDragRelease() {
        std::cout << "----------edge picking: drag release----------" << std::endl;
        std::cout << "multiple picking: " << this->varToggleMultiSelect()->getValue() << std::endl;
        std::cout << "Interation radius:" << this->varInteractionRadius()->getValue() << std::endl;
        std::cout << "selected num/ total num:" << this->tempNumS << "/" << this->tempNumT << std::endl;
    }

    template <typename TDataType>
    void EdgeInteraction<TDataType>::calcIntersectClick() {
        if (this->varTogglePicker()->getData()) calcEdgeIntersectClick();
    }

    template <typename TDataType>
    void EdgeInteraction<TDataType>::calcIntersectDrag() {
        if (this->varTogglePicker()->getData()) calcEdgeIntersectDrag();
    }

    DEFINE_CLASS(EdgeInteraction);
} // namespace dyno