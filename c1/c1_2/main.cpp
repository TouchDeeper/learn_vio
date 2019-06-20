#include <iostream>
#include <Eigen/Core>
// Eigen 几何模块
#include <Eigen/Geometry>
#include "sophus/so3.h"

int main() {
    Eigen::Vector3d euler_angle(1,3,0.5);//ZYX 欧拉角
    Eigen::Matrix3d R = (Eigen::AngleAxisd(euler_angle[0],Eigen::Vector3d::UnitZ())*Eigen::AngleAxisd(euler_angle[1],Eigen::Vector3d::UnitY())*Eigen::AngleAxisd(euler_angle[2],Eigen::Vector3d::UnitX())).matrix();
    Eigen::Quaterniond q(R);
    Eigen::Vector3d update(0.01, 0.02, 0.03);

    Sophus::SO3 SO3_R(R);
    Sophus::SO3 SO3_new_R = SO3_R * Sophus::SO3::exp(update);
    Eigen::Matrix3d new_R = SO3_new_R.matrix();

    Eigen::Quaterniond update_quat;
    update_quat.w() = 1;
    update_quat.x() = 0.5*update[0];
    update_quat.y() = 0.5*update[1];
    update_quat.z() = 0.5*update[2];
    // update the q, then normalize the q
    Eigen::Quaterniond new_q = q * update_quat;
    std::cout<<"new_q = \n"<<new_q.coeffs()<<std::endl;
    new_q.normalize();
    std::cout<<"new_q after normalized = \n"<<new_q.coeffs()<<std::endl;
    // normalize the update_quat, then update the q
    update_quat.normalize();
    Eigen::Quaterniond new_q_normalize = q * update_quat;
    std::cout<<"new_q after update_quat normalized = \n"<<new_q_normalize.coeffs()<<std::endl;

    Eigen::Quaterniond new_R_quat(new_R);
    std::cout<<"new_R_quat = \n"<<new_R_quat.coeffs()<<std::endl;

    //check the error in two way
    Eigen::Quaterniond error_update_normalized = new_R_quat.inverse() * new_q;
    Eigen::Quaterniond error_q_normalized = new_R_quat.inverse() * new_q_normalize;
    if(error_update_normalized.norm() > error_q_normalized.norm())
        std::cout<<"normalize the update_quat, then update the q is more close"<<std::endl;
    if(error_update_normalized.norm() < error_q_normalized.norm())
        std::cout<<"update the q, then normalize the q is more close"<<std::endl;
    else
        std:: cout<<"the result of two ways above is the same "<<std::endl;

    //check the new_R_quat * new_q_normalize.inverse() = I?
    Eigen::Quaterniond error_R_q = new_R_quat * new_q_normalize.inverse();
    std::cout<<"error_R_q = \n"<<error_R_q.coeffs()<<std::endl;
}