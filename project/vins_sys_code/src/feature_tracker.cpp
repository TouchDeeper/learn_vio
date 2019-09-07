#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}
/**
 * @brief   对跟踪点进行排序并去除密集点
 * @Description 对跟踪到的特征点，按照被追踪到的次数排序并依次选点
 *              使用mask进行类似非极大抑制，半径为30，去掉密集点，使特征点分布均匀
 * @return      void
*/
void FeatureTracker::setMask()
{
    std::cout<<"start set mask"<<std::endl;
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
    {
//        std::cout<<"                                 ids[i] = "<<ids[i]<<std::endl;
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));
    }

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}
/**
 * @brief   对图像使用光流法进行特征点跟踪
 * @Description createCLAHE() 对图像进行自适应直方图均衡化
 *              calcOpticalFlowPyrLK() LK金字塔光流法
 *              setMask() 对跟踪点进行排序，设置mask
 *              rejectWithF() 通过基本矩阵剔除outliers
 *              goodFeaturesToTrack() 添加特征点(shi-tomasi角点)，确保每帧都有足够的特征点
 *              addPoints()添加新的追踪点
 *              undistortedPoints() 对角点图像坐标去畸变矫正，并计算每个角点的速度
 * @param[in]   _img 输入图像
 * @param[in]   _cur_time 当前时间（图像时间戳）
 * @return      void
*/
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;
    //如果EQUALIZE=1，表示太亮或太暗，进行直方图均衡化处理
    if (EQUALIZE) //针对比较暗的场景，应用后可以提取到更多的特征点
    {
        //自适应直方图均衡
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        //ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    if (forw_img.empty())
    {
        //如果当前帧的图像数据forw_img为空，说明当前是第一次读入图像数据
        //将读入的图像赋给当前帧forw_img，同时还赋给prev_img、cur_img
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        //否则，说明之前就已经有图像读入，只需要更新当前帧forw_img的数据
        forw_img = img;
    }
    //此时forw_pts还保存的是上一帧图像中的特征点，所以把它清除
    forw_pts.clear();

    if (cur_pts.size() > 0)
    {
        std::cout<<"start calculate lk optical flow"<<std::endl;
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        //调用cv::calcOpticalFlowPyrLK()对前一帧的特征点cur_pts进行LK金字塔光流跟踪，得到forw_pts
        //status标记了从前一帧cur_img到forw_img特征点的跟踪状态，无法被追踪到的点标记为0
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);
        //将位于图像边界外的点标记为0
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        //根据status,把跟踪失败的点剔除
        //不仅要从当前帧数据forw_pts中剔除，而且还要从cur_un_pts、prev_pts和cur_pts中剔除
        //prev_pts和cur_pts中的特征点是一一对应的
        //记录特征点id的ids，和记录特征点被跟踪次数的track_cnt也要剔除
        //这里跟踪到的特征点会变少，后面会进行补充
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        //ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
        printf("temporal optical flow costs: %fms", t_o.toc());
    }
    //光流追踪成功,特征点被成功跟踪的次数就加1
    //数值代表被追踪的次数，数值越大，说明被追踪的就越久
    for (auto &n : track_cnt)
        n++;

    if (PUB_THIS_FRAME)
    {
        //通过基本矩阵剔除outliers
        rejectWithF();
        std::cout<<"set mask begins"<<std::endl;
        //ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();//保证相邻的特征点之间要相隔30个像素,设置mask
        //ROS_DEBUG("set mask costs %fms", t_m.toc());

        //ROS_DEBUG("detect feature begins");
        TicToc t_t;
        //计算是否需要提取新的特征点
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            std::cout<<"start detect corners"<<std::endl;
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);//寻找更多的特征点
        }
        else
            n_pts.clear();
        //ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        //ROS_DEBUG("add feature begins");
        TicToc t_a;
        //添将新检测到的特征点n_pts添加到forw_pts中，id初始化-1,track_cnt初始化为1.
        addPoints();
        //ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    //当下一帧图像到来时，当前帧数据就成为了上一帧发布的数据
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    //把当前帧的数据forw_img、forw_pts赋给上一帧cur_img、cur_pts
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();//这里不仅去畸变，还计算了特征点在归一化平面上的移动速度
    prev_time = cur_time;
}
/**
 * 读取特征，并计算特征速度
 * @param features_norm_plane 归一化平面上的特征
 * @param _cur_time
 */
void FeatureTracker::readFeatures(const std::vector<cv::Point2f>  &features_norm_plane, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

//    if (EQUALIZE) //针对比较暗的场景，应用后可以提取到更多的特征点
//    {
//        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
//        TicToc t_c;
//        clahe->apply(_img, img);
//        //ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
//    }
//    else
//        img = _img;

//    if (forw_img.empty())
//    {
//        prev_img = cur_img = forw_img = img;
//    }
//    else
//    {
//        forw_img = img;
//    }
//
//    forw_pts.clear();

    //将归一化平面的坐标转换成图像平面的坐标
    std::vector<cv::Point2f>  features;//图像平面的特征点坐标
    for (int i = 0; i < features_norm_plane.size(); ++i) {
        if(CAM_WITH_NOISE)
            features.emplace_back(cv::Point2f(features_norm_plane[i].x, features_norm_plane[i].y));//喂的是像素平面的特征
        else{
            //喂的是归一化平面的特征
            Eigen::Vector2d feature_norm_plane_tmp(features_norm_plane[i].x, features_norm_plane[i].y);
            Eigen::Vector2d feature_tmp;
            m_camera->undistToPlane(feature_norm_plane_tmp, feature_tmp);
            features.emplace_back(cv::Point2f(feature_tmp(0), feature_tmp(1)));
        }




    }


//    std::cout<<"features_norm_plane[3] :"<<std::endl<<features_norm_plane[3]<<std::endl;

//    if (forw_pts.empty())
//    {
//        prev_pts = cur_pts = forw_pts = features;
//    }
//    else
//    {
//        forw_pts = features;
//    }
    //此时forw_pts还保存的是上一帧图像中的特征点，所以把它清除
    forw_pts.clear();


    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        //将新进来的特征传递给forw_pts
        forw_pts = features;
//        vector<uchar> status;
//        vector<float> err;
//        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);
//
//        for (int i = 0; i < int(forw_pts.size()); i++)
//            if (status[i] && !inBorder(forw_pts[i]))
//                status[i] = 0;
        //这里跟踪到的特征点会变少，后面会进行补充
//        status.resize(forw_pts.size(),1);
//        reduceVector(prev_pts, status);
//        reduceVector(cur_pts, status);
//        reduceVector(forw_pts, status);
//        reduceVector(ids, status);
//        reduceVector(cur_un_pts, status);
//        reduceVector(track_cnt, status);
        //ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt)
        n++;

    if (PUB_THIS_FRAME)
    {
//        rejectWithF();
        //ROS_DEBUG("set mask begins");
        TicToc t_m;
//        setMask();
        //ROS_DEBUG("set mask costs %fms", t_m.toc());

        //ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)//特征恒定，只有在第一次PUB_THIS_FRAME为TRUE时候才会发生
        {
//            std::cout<<"add features, MAX_CNT = "<<MAX_CNT<<"  forw.pts.size = "<<forw_pts.size()<<std::endl;
//            if(mask.empty())
//                cout << "mask is empty " << endl;
//            if (mask.type() != CV_8UC1)
//                cout << "mask type wrong " << endl;
//            if (mask.size() != forw_img.size())
//                cout << "wrong size " << endl;
//            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);//寻找更多的特征点
            n_pts = features;
        }
        else
            n_pts.clear();
        //ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        //ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        //ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
//    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();//这里不仅去畸变，还计算了特征点在归一化平面上的移动速度
    prev_time = cur_time;
}
/**
 * @brief   通过F矩阵去除outliers
 * @Description 将图像坐标转换为归一化坐标
 *              cv::findFundamentalMat()计算F矩阵
 *              reduceVector()去除outliers
 * @return      void
*/
void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        //ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        //ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        //ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    cout << "reading paramerter of camera " << calib_file << endl;
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a_.x(), a_.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
//        std::cout<<"liftProjective "<<std::endl;
        m_camera->liftProjective(a, b);//这里进行了去畸变
//        std::cout<<"liftProjective done "<<std::endl;
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));

        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }

    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
