#include "PointInteraction.h"
#include <iostream>
#include <OrbitCamera.h>

// #include <thrust/sort.h>

namespace dyno
{
    template <typename TDataType>
    PointInteraction<TDataType>::PointInteraction() {
        this->ray1 = TRay3D<Real>();
        this->ray2 = TRay3D<Real>();
        this->isPressed = false;

        this->outOtherPointSet()->setDataPtr(std::make_shared<PointSet3f>());
        this->outOtherPointSet()->getDataPtr()->getPoints().resize(0);
        this->outSelectedPointSet()->setDataPtr(std::make_shared<PointSet3f>());
        this->outSelectedPointSet()->getDataPtr()->getPoints().resize(0);

        this->addKernel("PointRay",
                        std::make_shared<VkProgram>(BUFFER(Coord), BUFFER(int), BUFFER(int), BUFFER(Real),
                                                    CONSTANT(uint), CONSTANT(TRay3D<Real>), CONSTANT(Real)));
        this->kernel("PointRay")
            ->load(getSpvFile("shaders/glsl/interaction/PointRay.comp.spv"));

        this->addKernel("Nearest",
                        std::make_shared<VkProgram>(BUFFER(int), BUFFER(int), CONSTANT(uint), CONSTANT(int)));
        this->kernel("Nearest")->load(VkSystem::instance()->getAssetPath() /
                                      "shaders/glsl/interaction/Nearest.comp.spv");

        this->addKernel("Merge", std::make_shared<VkProgram>(BUFFER(int), BUFFER(int), BUFFER(int), BUFFER(int),
                                                                  CONSTANT(uint), CONSTANT(int)));
        this->kernel("Merge")
            ->load(getSpvFile("shaders/glsl/interaction/Merge.comp.spv"));

        this->addKernel("PointAssignOut",
                        std::make_shared<VkProgram>(BUFFER(Coord), BUFFER(Coord), BUFFER(Coord), BUFFER(int),
                                                    BUFFER(int), BUFFER(int), BUFFER(int), CONSTANT(uint)));
        this->kernel("PointAssignOut")
            ->load(getSpvFile("shaders/glsl/interaction/PointAssignOut.comp.spv"));

        this->addKernel("PointBox", std::make_shared<VkProgram>(BUFFER(Coord), BUFFER(int), BUFFER(int), CONSTANT(uint),
                                                                CONSTANT(TPlane3D<Real>), CONSTANT(TPlane3D<Real>),
                                                                CONSTANT(TPlane3D<Real>), CONSTANT(TPlane3D<Real>),
                                                                CONSTANT(TRay3D<Real>), CONSTANT(Real)));
        this->kernel("PointBox")
            ->load(getSpvFile("shaders/glsl/interaction/PointBox.comp.spv"));
    }

    template <typename TDataType>
    PointInteraction<TDataType>::~PointInteraction() {
        this->pointIntersectedIndex.clear();
        this->tempPointIntersectedIndex.clear();
    }

    template <typename TDataType>
    void PointInteraction<TDataType>::onEvent(PMouseEvent event) {
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
                if (this->varPointPickingType()->getValue() == PickingTypeSelection::Both ||
                    this->varPointPickingType()->getValue() == PickingTypeSelection::Click)
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
                        if (this->varPointPickingType()->getValue() == PickingTypeSelection::Both ||
                            this->varPointPickingType()->getValue() == PickingTypeSelection::Click)
                            this->calcIntersectClick();
                    }
                    else {
                        if (this->varPointPickingType()->getValue() == PickingTypeSelection::Both ||
                            this->varPointPickingType()->getValue() == PickingTypeSelection::Drag)
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
    void PointInteraction<TDataType>::calcPointIntersectClick() {
        VkCompContext::Holder holder;
        holder.delaySubmit(true);

        PointSet3f initialPointSet = this->inInitialPointSet()->getData();
        DArray<Coord> points = initialPointSet.getPoints();

        DArray<int> intersected(points.size());
        DArray<int> unintersected(points.size());
        this->tempNumT = points.size();

        DArray<Real> pointDistance;
        pointDistance.resize(points.size());
        VkConstant<uint> vk_num {points.size()};
        VkConstant<TRay3D<Real>> vk_ray {this->ray1};
        VkConstant<Real> vk_radius {this->varInteractionRadius()->getData()};
        this->kernel("PointRay")
            ->submit(vkDispatchSize(vk_num, 64), points.handle(), intersected.handle(), unintersected.handle(),
                    pointDistance.handle(), &vk_num, &vk_ray, &vk_radius);

        vk_num.setValue(intersected.size());
        VkConstant<int> vk_min_index {(int)mMin.reduce(*pointDistance.handle())};
        this->kernel("Nearest")->submit(vkDispatchSize(vk_num, 64), intersected.handle(), unintersected.handle(),
                                       &vk_num, &vk_min_index);

        this->tempPointIntersectedIndex.assign(intersected);

        if (this->varToggleMultiSelect()->getData()) {
            if (this->pointIntersectedIndex.size() == 0) {
                this->pointIntersectedIndex.resize(points.size());
                vkFill(*points.handle(), 0);
            }
            DArray<int> outIntersected;
            outIntersected.resize(intersected.size());
            DArray<int> outUnintersected;
            outUnintersected.resize(unintersected.size());
            VkConstant<int> vk_select_type {(int)this->varMultiSelectionType()->getValue().currentKey()};
            this->kernel("Merge")
                ->submit(vkDispatchSize(vk_num, 64), this->pointIntersectedIndex.handle(), intersected.handle(),
                        outIntersected.handle(), outUnintersected.handle(), &vk_num, &vk_select_type);

            intersected.assign(outIntersected);
            unintersected.assign(outUnintersected);
        }
        else {
            this->pointIntersectedIndex.assign(intersected);
        }
        DArray<int> intersected_o;
        intersected_o.assign(intersected);

        int intersected_size = mReduce.reduce(*intersected.handle());
        DArray<int> outPointIndex(intersected_size);
        mScan.scan(*intersected.handle(), *intersected.handle(), VkScan<int>::Exclusive);
        DArray<Coord> intersected_points(intersected_size);

        int unintersected_size = mReduce.reduce(*unintersected.handle());
        mScan.scan(*unintersected.handle(), *unintersected.handle(), VkScan<int>::Exclusive);
        DArray<Coord> unintersected_points(unintersected_size);

        vk_num.setValue(points.size());
        this->kernel("PointAssignOut")
            ->submit(vkDispatchSize(vk_num, 64), points.handle(), intersected_points.handle(),
                    unintersected_points.handle(), outPointIndex.handle(), intersected.handle(), unintersected.handle(),
                    intersected_o.handle(), &vk_num);

        this->tempNumS = intersected_size;
        this->outSelectedPointSet()->getDataPtr()->copyFrom(initialPointSet);
        this->outSelectedPointSet()->getDataPtr()->setPoints(intersected_points);
        this->outOtherPointSet()->getDataPtr()->copyFrom(initialPointSet);
        this->outOtherPointSet()->getDataPtr()->setPoints(unintersected_points);
        if (this->varToggleIndexOutput()->getValue()) {
            this->outPointIndex()->getDataPtr()->assign(outPointIndex);
        }
        else {
            this->outPointIndex()->getDataPtr()->assign(intersected_o);
        }
    }

    template <typename TDataType>
    void PointInteraction<TDataType>::calcPointIntersectDrag() {
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

        PointSet3f initialPointSet = this->inInitialPointSet()->getData();
        DArray<Coord> points = initialPointSet.getPoints();
        DArray<int> intersected(points.size());

        DArray<int> unintersected(points.size());
        this->tempNumT = points.size();

        VkConstant<uint> vk_num {points.size()};
        VkConstant<TRay3D<Real>> vk_ray {this->ray1};
        VkConstant<Real> vk_radius {this->varInteractionRadius()->getValue()};
        this->kernel("PointBox")
            ->submit(vkDispatchSize(vk_num, 64), points.handle(), intersected.handle(), unintersected.handle(), &vk_num,
                    &plane13, &plane42, &plane14, &plane32, &vk_ray, &vk_radius);

        this->tempPointIntersectedIndex.assign(intersected);

        if (this->varToggleMultiSelect()->getData()) {
            if (this->pointIntersectedIndex.size() == 0) {
                this->pointIntersectedIndex.resize(points.size());
            }
            DArray<int> outIntersected(intersected.size());
            DArray<int> outUnintersected(unintersected.size());
            VkConstant<int> vk_select_type {(int)this->varMultiSelectionType()->getValue().currentKey()};
            this->kernel("Merge")
                ->submit(vkDispatchSize(vk_num, 64), this->pointIntersectedIndex.handle(), intersected.handle(),
                        outIntersected.handle(), outUnintersected.handle(), &vk_num, &vk_select_type);

            intersected.assign(outIntersected);
            unintersected.assign(outUnintersected);
        }
        else {
            this->pointIntersectedIndex.assign(intersected);
        }

        DArray<int> intersected_o;
        intersected_o.assign(intersected);

        int intersected_size = mReduce.reduce(*intersected.handle());
        DArray<int> outPointIndex(intersected_size);
        mScan.scan(*intersected.handle(), *intersected.handle(), VkScan<int>::Exclusive);
        DArray<Coord> intersected_points(intersected_size);

        int unintersected_size = mReduce.reduce(*unintersected.handle());
        mScan.scan(*unintersected.handle(), *unintersected.handle(), VkScan<int>::Exclusive);
        DArray<Coord> unintersected_points(unintersected_size);

        this->kernel("PointAssignOut")
            ->submit(vkDispatchSize(vk_num, 64), points.handle(), intersected_points.handle(),
                    unintersected_points.handle(), outPointIndex.handle(), intersected.handle(), unintersected.handle(),
                    intersected_o.handle(), &vk_num);

        this->tempNumS = intersected_size;
        this->outSelectedPointSet()->getDataPtr()->copyFrom(initialPointSet);
        this->outSelectedPointSet()->getDataPtr()->setPoints(intersected_points);
        this->outOtherPointSet()->getDataPtr()->copyFrom(initialPointSet);
        this->outOtherPointSet()->getDataPtr()->setPoints(unintersected_points);
        if (this->varToggleIndexOutput()->getValue()) {
            this->outPointIndex()->getDataPtr()->assign(outPointIndex);
        }
        else {
            this->outPointIndex()->getDataPtr()->assign(intersected_o);
        }
    }

    template <typename TDataType>
    void PointInteraction<TDataType>::mergeIndex() {
        VkCompContext::Holder holder;
        holder.delaySubmit(true);

		PointSet3f initialPointSet = this->inInitialPointSet()->getData();
		DArray<Coord> points = initialPointSet.getPoints();
		DArray<int> intersected(points.size());

		DArray<int> unintersected(points.size());
		this->tempNumT = points.size();

		DArray<int> outIntersected(intersected.size());
		DArray<int> outUnintersected(unintersected.size());

        VkConstant<uint> vk_num{ this->pointIntersectedIndex.size() };

        if (this->varToggleMultiSelect()->getData()) {
            VkConstant<int> vk_select_type{ (int)this->varMultiSelectionType()->getValue().currentKey() };
            this->kernel("Merge")
                ->submit(vkDispatchSize(vk_num, 64), this->pointIntersectedIndex.handle(), this->tempPointIntersectedIndex.handle(),
                    outIntersected.handle(), outUnintersected.handle(), &vk_num, &vk_select_type);
        }

		intersected.assign(outIntersected);
		unintersected.assign(outUnintersected);
		this->pointIntersectedIndex.assign(intersected);

        DArray<int> intersected_o;
        intersected_o.assign(intersected);

        int intersected_size = mReduce.reduce(*intersected.handle());
        DArray<int> outPointIndex(intersected_size);
        mScan.scan(*intersected.handle(), *intersected.handle(), VkScan<int>::Exclusive);
        DArray<Coord> intersected_points(intersected_size);

        int unintersected_size = mReduce.reduce(*unintersected.handle());
        mScan.scan(*unintersected.handle(), *unintersected.handle(), VkScan<int>::Exclusive);
        DArray<Coord> unintersected_points(unintersected_size);

        this->kernel("PointAssignOut")
            ->submit(vkDispatchSize(vk_num, 64), points.handle(), intersected_points.handle(),
                unintersected_points.handle(), outPointIndex.handle(), intersected.handle(), unintersected.handle(),
                intersected_o.handle(), &vk_num);

		this->tempNumS = intersected_size;
		this->outSelectedPointSet()->getDataPtr()->copyFrom(initialPointSet);
		this->outSelectedPointSet()->getDataPtr()->setPoints(intersected_points);
		this->outOtherPointSet()->getDataPtr()->copyFrom(initialPointSet);
		this->outOtherPointSet()->getDataPtr()->setPoints(unintersected_points);
		if (this->varToggleIndexOutput()->getValue())
		{
			this->outPointIndex()->getDataPtr()->assign(outPointIndex);
		}
		else
		{
			this->outPointIndex()->getDataPtr()->assign(intersected_o);
		}
    }

    template <typename TDataType>
    void PointInteraction<TDataType>::printInfoClick() {
        std::cout << "----------point picking: click----------" << std::endl;
        std::cout << "multiple picking: " << this->varToggleMultiSelect()->getValue() << std::endl;
        std::cout << "Interation radius:" << this->varInteractionRadius()->getValue() << std::endl;
        std::cout << "selected num/ total num:" << this->tempNumS << "/" << this->tempNumT << std::endl;
    }

    template <typename TDataType>
    void PointInteraction<TDataType>::printInfoDragging() {
        std::cout << "----------point picking: dragging----------" << std::endl;
        std::cout << "multiple picking: " << this->varToggleMultiSelect()->getValue() << std::endl;
        std::cout << "Interation radius:" << this->varInteractionRadius()->getValue() << std::endl;
        std::cout << "selected num/ total num:" << this->tempNumS << "/" << this->tempNumT << std::endl;
    }

    template <typename TDataType>
    void PointInteraction<TDataType>::printInfoDragRelease() {
        std::cout << "----------point picking: drag release----------" << std::endl;
        std::cout << "multiple picking: " << this->varToggleMultiSelect()->getValue() << std::endl;
        std::cout << "Interation radius:" << this->varInteractionRadius()->getValue() << std::endl;
        std::cout << "selected num/ total num:" << this->tempNumS << "/" << this->tempNumT << std::endl;
    }

    template <typename TDataType>
    void PointInteraction<TDataType>::calcIntersectClick() {
        if (this->varTogglePicker()->getData()) calcPointIntersectClick();
    }

    template <typename TDataType>
    void PointInteraction<TDataType>::calcIntersectDrag() {
        if (this->varTogglePicker()->getData()) calcPointIntersectDrag();
    }

    DEFINE_CLASS(PointInteraction);
} // namespace dyno