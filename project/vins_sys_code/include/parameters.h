#pragma once

// #include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "utility/utility.h"
// #include <opencv2/opencv.hpp>
// #include <opencv2/core/eigen.hpp>
#include <fstream>

//#ifndef USE_OPENMP
//#define USE_OPENMP 1
//#endif

extern int RUN_COUNT;
extern int RUN_NUM;
extern std::vector <std::string> LOOP_PARAMETER;
extern std::string NOW_LOOP;
extern int VERBOSE;
extern int STOP_REASON;//是否输出每次迭代的停止原因；
extern int HDL_CHOOSE; //是否输出hdl选择过程
extern int RADIUS_CHI_G_OUTPUT; //是否输出dog-leg优化过程中的radius, chi,g
extern std::string sConfig_file;
extern double stopThresholdGradient; // 迭代退出gradient阈值条件
extern int SOLVER_TYPE; //选择solver type (LM, DOGLEG)
extern int OPTIMIZE_LM;
extern int JACOBIAN_SCALING;
extern int DTD_SCALING;
extern double T_HESSIAN_ALL;
extern double T_SOLVE_COST_ALL;
extern int NUM_MAKE_HESSIAN;
extern int USE_OPENMP_THREADS;
extern int NEW_LM_UPDATE;
extern int SHOW_LAMBDA;
extern enum enum_SOLVER_TYPE{LM, DOGLEG, HYBRID} k_SOLVER_TYPE;
//feature tracker
// extern int ROW;
// extern int COL;
const int NUM_OF_CAM = 1;

extern int FOCAL_LENGTH;
extern std::string IMAGE_TOPIC;
extern std::string IMU_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
// extern int WINDOW_SIZE;
extern int FREQ;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern bool STEREO_TRACK;
extern int EQUALIZE;
extern int FISHEYE;
extern bool PUB_THIS_FRAME;

//estimator

// const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
// const int NUM_OF_CAM = 1;
const int NUM_OF_F = 1000;
//#define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string IMU_TOPIC;
extern double TD;
extern double TR;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern double ROW, COL;

// void readParameters(ros::NodeHandle &n);

void readParameters(std::string config_file);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
