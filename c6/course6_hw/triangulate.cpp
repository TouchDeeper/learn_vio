//
// Created by hyj on 18-11-11.
//
#include <iostream>
#include <vector>
#include <random>  
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

struct Pose
{
    Pose(Eigen::Matrix3d R, Eigen::Vector3d t):Rwc(R),qwc(R),twc(t) {};
    Eigen::Matrix3d Rwc;
    Eigen::Quaterniond qwc;
    Eigen::Vector3d twc;

    Eigen::Vector2d uv;    // 这帧图像观测到的特征坐标
};
int main()
{

    int poseNums = 10;
    double radius = 8;
    double fx = 1.;
    double fy = 1.;
    std::vector<Pose> camera_pose;
    for(int n = 0; n < poseNums; ++n ) {
        double theta = n * 2 * M_PI / ( poseNums * 4); // 1/4 圆弧
        // 绕 z轴 旋转
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        camera_pose.push_back(Pose(R,t));
    }

    // 随机数生成 1 个 三维特征点
    std::default_random_engine generator;
    std::uniform_real_distribution<double> xy_rand(-4, 4.0);
    std::uniform_real_distribution<double> z_rand(8., 10.);
    double tx = xy_rand(generator);
    double ty = xy_rand(generator);
    double tz = z_rand(generator);

    Eigen::Vector3d Pw(tx, ty, tz);
    // 这个特征从第三帧相机开始被观测，i=3
    int start_frame_id = 3;
    int end_frame_id = poseNums;
    for (int i = start_frame_id; i < end_frame_id; ++i) {
        Eigen::Matrix3d Rcw = camera_pose[i].Rwc.transpose();
        Eigen::Vector3d Pc = Rcw * (Pw - camera_pose[i].twc);

        double x = Pc.x();
        double y = Pc.y();
        double z = Pc.z();

        camera_pose[i].uv = Eigen::Vector2d(x/z,y/z);
    }
    
    /// TODO::homework; 请完成三角化估计深度的代码
    // 遍历所有的观测数据，并三角化
    Eigen::Vector3d P_est;           // 结果保存到这个变量
    P_est.setZero();

    /* your code begin */
    int num_sense = end_frame_id - start_frame_id + 1; //计算观测数据个数
    int num_rows = 2 * num_sense; //compute the rows of the matrix D
    std::cout<<"num_rows = "<<num_rows<<std::endl;
    Eigen::MatrixXd D;
    D.resize(num_rows,4);
    for (int j = start_frame_id; j < end_frame_id; ++j) {
        Eigen::Isometry3d w_P_c = Eigen::Isometry3d::Identity();
        w_P_c.rotate(camera_pose[j].Rwc);
        w_P_c.pretranslate(camera_pose[j].twc);
        Eigen::Isometry3d c_P_w = Eigen::Isometry3d::Identity();
        c_P_w = w_P_c.inverse();
        int row_index = 2 * ( j - start_frame_id );
        D.block(row_index,0,1,4) = camera_pose[j].uv[0] * c_P_w.matrix().row(2) - c_P_w.matrix().row(0);
        D.block(row_index+1,0,1,4) = camera_pose[j].uv[1] * c_P_w.matrix().row(2) - c_P_w.matrix().row(1);
    }
    //矩阵缩放，防止数值问题
    double max_D = D.maxCoeff();
    Eigen::Matrix4d S = 1 / max_D * Eigen::Matrix4d::Identity();
    D = D * S;

    //svd
    Eigen::Matrix4d DTD = D.transpose()*D;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(DTD, Eigen::ComputeThinU | Eigen::ComputeThinV );
    Eigen::Matrix4d V = svd.matrixV(), U = svd.matrixU();
    Eigen::Matrix4d  eigenvalue_2 = U.inverse() * DTD * V.transpose().inverse(); // compute the eigenvalue S = U^-1 * A * VT * -1
    if(eigenvalue_2(3,3) > eigenvalue_2(2,2) * 1e-14)
    {
        std::cout<<"triangularization invalid"<<std::endl;
        return 0;
    }
//    std::cout<<"eigenvalue_2 :\n"<<eigenvalue_2<<std::endl;
    Eigen::Vector4d P_est_S = S.inverse() * U.col(3);
    P_est(0) = P_est_S(0)/P_est_S(3);
    P_est(1) = P_est_S(1)/P_est_S(3);
    P_est(2) = P_est_S(2)/P_est_S(3);



    /* your code end */
    
    std::cout <<"ground truth: \n"<< Pw.transpose() <<std::endl;
    std::cout <<"your result: \n"<< P_est.transpose() <<std::endl;
    return 0;
}
