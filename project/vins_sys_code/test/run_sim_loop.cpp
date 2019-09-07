
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <thread>
#include <iomanip>

#include <cv.h>
#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <eigen3/Eigen/Dense>
#include "System.h"

using namespace std;
using namespace cv;
using namespace Eigen;

const int nDelayTimes = 2;
//string sData_path = "/media/wang/File/dataset/EuRoc/MH-05/mav0/";
string sConfig_path = "../config/sim/";

std::shared_ptr<System> pSystem;

void PubImuData()
{
	string sImu_data_file = sConfig_path + "data/imu_pose_noise.txt";//记得改config文件
//	string sImu_data_file = sConfig_path + "data/imu_pose.txt";
	cout << "1 PubImuData start sImu_data_filea: " << sImu_data_file << endl;
	ifstream fsImu;
	fsImu.open(sImu_data_file.c_str());
	if (!fsImu.is_open())
	{
		cerr << "Failed to open imu file! " << sImu_data_file << endl;
		return;
	}

	std::string sImu_line;
	double dStampNSec = 0.0;
	Vector3d vAcc;
	Vector3d vGyr;
	while (std::getline(fsImu, sImu_line) && !sImu_line.empty()) // read imu data
	{
		std::istringstream ssImuData(sImu_line);
		Eigen::Quaterniond q;
		Eigen::Vector3d t;
		ssImuData >> dStampNSec >> q.w() >> q.x() >> q.y() >> q.z() >> t(0) >> t(1) >> t(2) >> vGyr.x() >> vGyr.y() >> vGyr.z() >> vAcc.x() >> vAcc.y() >> vAcc.z();
		// cout << "Imu t: " << fixed << dStampNSec << " gyr: " << vGyr.transpose() << " acc: " << vAcc.transpose() << endl;
		pSystem->PubImuData(dStampNSec, vGyr, vAcc);
		usleep(5000*nDelayTimes);
	}
	fsImu.close();
}

void PubImageData()
{
    int i = 0;
    while (true) {
        string sImage_file;
        if(CAM_WITH_NOISE)
            sImage_file = sConfig_path + "data/keyframe/pixel_all_points_" + std::to_string(i) + ".txt";
        else
            sImage_file = sConfig_path + "data/keyframe/all_points_" + std::to_string(i) + ".txt";

        cout << "1 PubImageData start sImage_file: " << sImage_file << endl;

        ifstream fsImage;
        fsImage.open(sImage_file.c_str());
        if (!fsImage.is_open()) {
            std::cout << "Failed to open image file! " << sImage_file << endl;
            pSystem->set_stop_flag();
            return;
        }

        std::string sImage_line;
        double dStampNSec;
        string sImgFileName;
        // cv::namedWindow("SOURCE IMAGE", CV_WINDOW_AUTOSIZE);
        std::vector<cv::Point2f> features;

        while (std::getline(fsImage, sImage_line) && !sImage_line.empty()) {
            std::istringstream ssImageData(sImage_line);
            cv::Point2f feature_tmp;
            Eigen::Vector4d point;//useless
            ssImageData >> point(0) >> point(1) >> point(2) >> point(3) >> feature_tmp.x >> feature_tmp.y >> dStampNSec;
            features.push_back(feature_tmp);
            // cout << "Image t : " << fixed << dStampNSec << " Name: " << sImgFileName << endl;
//            string imagePath = sData_path + "cam0/data/" + sImgFileName;
//
//            Mat img = imread(imagePath.c_str(), 0);
//            if (img.empty()) {
//                cerr << "image is empty! path: " << imagePath << endl;
//                return;
//            }
        }
        pSystem->PubImageData(dStampNSec, features);
        // cv::imshow("SOURCE IMAGE", img);
        // cv::waitKey(0);
        usleep(50000 * nDelayTimes);
        i++;
        fsImage.close();
    }

}


int main(int argc, char **argv)
{
//	if(argc != 3)
//	{
//		cerr << "./run_euroc PATH_TO_FOLDER/MH-05/mav0 PATH_TO_CONFIG/config \n"
//			<< "For example: ./run_euroc /home/stevencui/dataset/EuRoC/MH-05/mav0/ ../config/"<< endl;
//		return -1;
//	}
//	sData_path = argv[1];
//	sConfig_path = argv[2];
    RUN_LOOP = true;
    RUN_NUM = 27;          // loop tims, must be odd
    LOOP_PARAMETER.emplace_back("ACC_N");
    LOOP_PARAMETER.emplace_back("GYR_N");
    LOOP_PARAMETER.emplace_back("ACC_W");
    LOOP_PARAMETER.emplace_back("GYR_W");
    for (int i = 0; i < LOOP_PARAMETER.size(); ++i) {
        NOW_LOOP = LOOP_PARAMETER[i];
        for ( RUN_COUNT = 0; RUN_COUNT < RUN_NUM; ++RUN_COUNT) {
            pSystem.reset(new System(sConfig_path,"sim"));
            // TODO remember to modify the ouput path
            std::thread thd_BackEnd(&System::ProcessBackEnd, pSystem);

            // sleep(5);
            std::thread thd_PubImuData(PubImuData);

            std::thread thd_PubImageData(PubImageData);

//            std::thread thd_Draw(&System::Draw, pSystem);

            thd_PubImuData.join();
            thd_PubImageData.join();

            thd_BackEnd.join();
//            thd_Draw.join();

            cout << "main end... see you ..." << endl;
        }
    }
	return 0;
}
